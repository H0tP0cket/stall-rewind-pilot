"""Build and validate v3 successors of the sealed v2 tasks (no model/provider calls).

v3 = v2 worker start + a fresh Git repository in W + the native SWE-smith
instance template. Native SWE-smith agents work in a Git checkout of the
instance branch (git diff/stash/checkout work there). v3 gives W a repository
whose single commit holds exactly the tracked start files; upstream history is
not imported, because it contains the "Bug Patch" commit (the clean code) and
the "Remove F2P Tests" commit (the withheld tests).

Per family: image v3 = the v1/v2 image + Debian ``git`` + system Git config;
the withheld-test needle scan and the v1 clean-source scan are re-run on it.
Per task: the v2 package is copied unchanged (W files, instructions, private
data); ``public/workspace-git.tar`` holds the frozen ``.git`` built by the
image's own git; the same qualification gates as v2 run again and are compared
with the v2 raw evidence; a Git check inside a real worker container proves the
start state and workflow; the MiningRunner fixture checks capture/restore.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tarfile
import zlib

from . import pipeline, pipeline_v2, source
from .adapter import file_manifest, sha256
from .adapter_v3 import (ADAPTER_ENTRY, GIT_MTIME, GIT_TAR, SCHEMA, START_MTIME, SWESmithLongV3TaskAdapter,
                         evaluator_sources, set_start_mtimes)
from .environment import LANE_LABEL, image_id
from .pipeline import CACHE, EVIDENCE, WORKTREE, write_json

PACKAGES_V2 = WORKTREE / "benchmark_tasks" / "packages" / "swesmith-long-v2"
PACKAGES = WORKTREE / "benchmark_tasks" / "packages" / "swesmith-long-v3"
ADMISSION_V2 = WORKTREE / "benchmark_tasks" / "admission" / "swesmith-long-v2"
ADMISSION = WORKTREE / "benchmark_tasks" / "admission" / "swesmith-long-v3"
EVIDENCE_V2 = EVIDENCE / "v2"
EVIDENCE_V3 = EVIDENCE / "v3"
WORK = CACHE / "work-v3"
HOLD_V2 = {"swesmith-python-pinyin-cbtf4sk7-v2": (
    "v2 HOLD carried forward: native 'Remove F2P Tests' deletes the source module pypinyin/core.py (FAIL_TO_PASS "
    "doctest ids), so a faithful native start and grade are not defined; see the v2 record")}

START_IDENTITY = {"name": "swesmith-long", "email": "start@swesmith-long.invalid"}
START_DATE = "2000-01-01T00:00:00+00:00"   # = adapter_v3.START_MTIME
START_MESSAGE = "Initial commit"            # the message of every SWE-smith mirror's root commit
BRANCH = "main"
WORKER_IDENTITY = {"name": "swesmith-long worker", "email": "worker@swesmith-long.invalid"}
SYSTEM_GIT_CONFIG = (
    ("user.name", WORKER_IDENTITY["name"]),
    ("user.email", WORKER_IDENTITY["email"]),
    # Restored W has new inodes/ctimes; compare only mtime (whole seconds), size
    # and mode so that git status neither re-hashes nor rewrites .git/index after
    # a start or a checkpoint restore (capture/restore keep mtime exactly).
    ("core.checkStat", "minimal"),
    ("core.trustctime", "false"),
)
INDEX_RULE = ("index v2 written from `git ls-tree -r -l HEAD`: entries sorted by path bytes; ctime = mtime = "
              f"{START_MTIME} s (0 ns); dev = ino = uid = gid = 0; mode and size from the tree; no extensions")
ADAPTATION = {
    "version": "swesmith-long.adaptation.v3",
    "predecessor": "swesmith-long.adaptation.v2",
    "changes": [
        "W is a fresh Git repository (branch main) with exactly one commit whose tree is the tracked worker start "
        "(v2 W minus the install-generated files, which the project's .gitignore already ignores). The FAIL_TO_PASS "
        "files are absent from the commit and from every Git object. No upstream history (native history holds "
        "the clean code in 'Bug Patch' and the withheld tests in 'Remove F2P Tests'). Fixed neutral identity "
        f"{START_IDENTITY['name']} <{START_IDENTITY['email']}>, date {START_DATE}, message '{START_MESSAGE}'.",
        f"Every start file has modification time {START_DATE} (the frozen index records it); .git entries one minute later.",
        "Worker image = the v1/v2 image + Debian git (no other change) + system Git config: worker identity "
        f"{WORKER_IDENTITY['name']} <{WORKER_IDENTITY['email']}>, core.checkStat=minimal, core.trustctime=false.",
        "Grading skips every .git path of the submitted W (Git never tracks such a path); otherwise v2's grader.",
        "Worker prompt: the native SWE-smith instance template verbatim (benchmark-task-prompt.v3; {{working_dir}} "
        "= /workspace); only the SWE-agent system template/tool mechanics are replaced by the shared mini-SWE harness.",
        "Unchanged from v2: withheld FAIL_TO_PASS files, native arm64 rebuild from official pins, network none, "
        "test-support restoration, stable-PASS_TO_PASS rule, documentation symlink copies (python-pinyin).",
    ],
    "unchanged": ["W file contents and instructions byte-identical to v2",
                  "bug patch, FAIL_TO_PASS/PASS_TO_PASS identities, score rule, test-support set, family split"],
}
# Installing git from the current bookworm archive requires perl (= its current
# version), whose essential perl-base must match; no Python file or library the
# tests load changes (pip freeze is compared, grades are compared with v2).
ALLOWED_DPKG_UPGRADES = {"perl-base": "required by git -> perl (= current bookworm version); not used by Python"}
DOCKERFILE = """FROM {base}
RUN apt-get update \\
 && apt-get install -y --no-install-recommends git \\
 && rm -rf /var/lib/apt/lists/* \\
{config} && git --version > /opt/swesmith-long/git-version.txt
LABEL trajectory-retention.lane="swesmith-long" trajectory-retention.swesmith-repo="{repo_key}" trajectory-retention.swesmith-long.image="v3"
"""


def v3_id(v2_id: str) -> str:
    if not v2_id.endswith("-v2"):
        raise ValueError(v2_id)
    return v2_id[:-3] + "-v3"


def _docker(*args: str, input_bytes: bytes | None = None, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], input=input_bytes, capture_output=True, timeout=timeout)


def _run_in_image(image: str, script: str, *, mounts: tuple[tuple[Path, str, bool], ...] = (), label: str = "probe",
                  timeout: int = 900) -> subprocess.CompletedProcess:
    name = f"tr-swesmith-long-v3-{label}-" + hashlib.sha256(os.urandom(8)).hexdigest()[:10]
    args = ["run", "--rm", "--name", name, "--label", LANE_LABEL, "--network", "none", "--entrypoint", "bash"]
    for src, dst, readonly in mounts:
        args += ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if readonly else "")]
    return _docker(*args, image, "-c", script, timeout=timeout)


# --------------------------------------------------------------------------- family image
def family_packages(key: str) -> list[str]:
    registry = json.loads((ADMISSION_V2 / "registry.json").read_text())
    ids = sorted(t["task_id"] for t in registry["tasks"] if t["task_id"].startswith("swesmith-" + pipeline.slug(key) + "-"))
    return ids + sorted(h for h in HOLD_V2 if h.startswith("swesmith-" + pipeline.slug(key) + "-"))


def build_family_image(key: str) -> dict:
    out = EVIDENCE_V3 / "families" / key
    out.mkdir(parents=True, exist_ok=True)
    v2_ids = [i for i in family_packages(key) if i not in HOLD_V2]
    v2_manifest = json.loads((PACKAGES_V2 / v2_ids[0] / "package.json").read_text())
    base_tag, base_id = v2_manifest["environment_tag"], v2_manifest["environment_image_id"]
    if image_id(base_tag) != base_id:
        raise RuntimeError(f"{base_tag} no longer resolves to the v2 image {base_id}")
    tag = base_tag.rsplit(":", 1)[0] + ":v3"
    context = CACHE / "build-v3" / key
    shutil.rmtree(context, ignore_errors=True)
    context.mkdir(parents=True)
    config = "".join(f" && git config --system {k} '{v}' \\\n" for k, v in SYSTEM_GIT_CONFIG)
    text = DOCKERFILE.format(base=base_tag, config=config, repo_key=key)
    (context / "Dockerfile").write_text(text)
    shutil.copy2(context / "Dockerfile", out / "Dockerfile")
    with (out / "build.log").open("wb") as log:
        result = subprocess.run(["docker", "build", "--platform", "linux/arm64", "--label", LANE_LABEL, "-t", tag,
                                 str(context)], stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if result.returncode:
        raise RuntimeError(f"v3 image build failed for {key}; see {out / 'build.log'}")
    new_id = image_id(tag)
    inspect = {i: json.loads(_docker("image", "inspect", i).stdout)[0] for i in (base_id, new_id)}
    base_layers, new_layers = (inspect[i]["RootFS"]["Layers"] for i in (base_id, new_id))
    probe = ("if command -v git >/dev/null; then git --version; else echo no-git; fi; echo '--config'; "
             "if command -v git >/dev/null; then git config --system --list; fi; echo '--dpkg'; "
             "dpkg-query -W -f='${Package}=${Version}\\n' | sort; echo '--pip'; python -m pip freeze --all | sort")

    def sections(image: str) -> dict:
        run = _run_in_image(image, probe, label="probe")
        if run.returncode:
            raise RuntimeError(f"image probe failed: {run.stderr.decode(errors='replace')[-500:]}")
        parts, current = {"git": []}, "git"
        for line in run.stdout.decode().splitlines():
            if line.startswith("--"):
                current = line[2:]
                parts[current] = []
            else:
                parts[current].append(line)
        return parts
    before, after = sections(base_id), sections(new_id)
    dpkg_before = dict(l.split("=", 1) for l in before["dpkg"])
    dpkg_after = dict(l.split("=", 1) for l in after["dpkg"])
    record = {"repo_key": key, "base_image_tag": base_tag, "base_image_id": base_id, "tag": tag, "image_id": new_id,
              "dockerfile_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "layers_extend_base": new_layers[:len(base_layers)] == base_layers,
              "added_layers": len(new_layers) - len(base_layers),
              "git_version": after["git"][0] if after["git"] else None,
              "system_git_config": after["config"],
              "system_git_config_expected": [f"{k.lower()}={v}" for k, v in SYSTEM_GIT_CONFIG],
              "dpkg_added": sorted(f"{p}={v}" for p, v in dpkg_after.items() if p not in dpkg_before),
              "dpkg_changed": sorted(f"{p}: {dpkg_before[p]} -> {v}" for p, v in dpkg_after.items()
                                     if p in dpkg_before and dpkg_before[p] != v),
              "dpkg_removed": sorted(p for p in dpkg_before if p not in dpkg_after),
              "pip_freeze_unchanged": before["pip"] == after["pip"], "pip_packages": len(after["pip"]),
              "base_had_git": before["git"] != ["no-git"]}
    record["dpkg_changed_allowed"] = ALLOWED_DPKG_UPGRADES
    record["image_ok"] = (record["layers_extend_base"] and record["added_layers"] == 1
                          and all(c.split(":", 1)[0] in ALLOWED_DPKG_UPGRADES for c in record["dpkg_changed"])
                          and not record["dpkg_removed"] and record["pip_freeze_unchanged"] and not record["base_had_git"]
                          and record["system_git_config"] == record["system_git_config_expected"])
    write_json(out / "image.json", record)
    return record


def family_scans(key: str, image: str) -> dict:
    """Re-run v2's withheld-test needle scan and v1's clean-source scan on the v3 image."""
    out = EVIDENCE_V3 / "families" / key
    shutil.copy2(EVIDENCE_V2 / "families" / key / "needles.json", out / "needles.json")
    here = Path(__file__).with_name("build_support")
    name = "tr-swesmith-long-v3-needles-" + hashlib.sha256(key.encode()).hexdigest()[:10]
    result = subprocess.run(["docker", "run", "--rm", "--name", name, "--label", LANE_LABEL, "--network", "none",
                             "--entrypoint", "python3", "--mount", f"type=bind,src={here},dst=/scan,readonly",
                             "--mount", f"type=bind,src={out},dst=/needles,readonly",
                             image, "-I", "/scan/scan_needles.py", "/needles/needles.json"],
                            check=True, capture_output=True, timeout=900)
    needles = {**json.loads(result.stdout), "image_id": image}
    write_json(out / "image-needles.json", needles)
    family = json.loads((EVIDENCE / "families" / key / "family.json").read_text())
    clean = CACHE / "build" / key / "clean"
    scan = pipeline.image_leak_scan(image, clean, family["project_top_levels"], out)
    write_json(out / "image-leak-scan.json", scan)
    return {"needles_files_scanned": needles["files_scanned"], "needle_hits": len(needles["hits"]),
            "clean_source_scan": {k: (len(v) if isinstance(v, list) else v) for k, v in scan.items() if k != "rule"}}


# --------------------------------------------------------------------------- git start
def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def write_index(tree_rows: list[tuple[int, str, str, int]], path: Path) -> None:
    """Deterministic index v2 for (mode, oid, path, size) rows (see INDEX_RULE)."""
    rows = sorted(tree_rows, key=lambda r: r[2].encode())
    body = bytearray(b"DIRC" + struct.pack(">II", 2, len(rows)))
    for mode, oid, name, size in rows:
        encoded = name.encode()
        entry = struct.pack(">10I", START_MTIME, 0, START_MTIME, 0, 0, 0, mode, 0, 0, size)
        entry += bytes.fromhex(oid) + struct.pack(">H", min(len(encoded), 0xFFF)) + encoded
        entry += b"\0" * (8 - len(entry) % 8)
        body += entry
    body += hashlib.sha1(bytes(body)).digest()
    path.write_bytes(bytes(body))


def deterministic_tar(entries: dict[str, tuple[int, bytes | None]], destination: Path) -> list[str]:
    """USTAR archive of name -> (mode, bytes or None for a directory): sorted, uid/gid 0, mtime GIT_MTIME."""
    names = sorted(entries)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in names:
            mode, data = entries[name]
            info = tarfile.TarInfo(name)
            info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode = GIT_MTIME, 0, 0, "", "", mode
            if data is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    destination.write_bytes(buffer.getvalue())
    return names


def read_container_git(raw_tar: Path) -> dict[str, tuple[int, bytes | None]]:
    entries = {}
    with tarfile.open(raw_tar) as archive:
        for member in archive.getmembers():
            name = member.name.rstrip("/")
            if not (name == ".git" or name.startswith(".git/")) or not (member.isdir() or member.isfile()):
                raise RuntimeError("unexpected .git entry " + member.name)
            entries[name] = (member.mode & 0o7777, None if member.isdir() else archive.extractfile(member).read())
    return entries


BUILD_SCRIPT = r"""set -eu
cp -r --preserve=mode,timestamps /stage/repo /build
cd /build
git init -q -b {branch}
git add -A
GIT_AUTHOR_NAME='{name}' GIT_AUTHOR_EMAIL='{email}' GIT_AUTHOR_DATE='{date}' \
GIT_COMMITTER_NAME='{name}' GIT_COMMITTER_EMAIL='{email}' GIT_COMMITTER_DATE='{date}' \
  git -c gc.auto=0 commit -q -m '{message}'
git rev-parse HEAD 'HEAD^{{tree}}' > /stage/ids
git ls-tree -r -l -z HEAD > /stage/ls-tree
git --version > /stage/git-version
rm .git/index
tar --numeric-owner -cf /stage/git-raw.tar .git
"""


def build_git_start(package: Path, image: str, work: Path, native_head: str, repo_key: str,
                    materialized: list[dict]) -> dict:
    stage = work / "git-build"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    source.copy_tree(package / "public/workspace", stage / "repo")
    set_start_mtimes(stage / "repo")
    script = BUILD_SCRIPT.format(branch=BRANCH, date=START_DATE, message=START_MESSAGE, **START_IDENTITY)
    run = _run_in_image(image, script, mounts=((stage, "/stage", False),), label="gitbuild")
    if run.returncode:
        raise RuntimeError("git start build failed: " + run.stderr.decode(errors="replace")[-1500:])
    commit, tree = (stage / "ids").read_text().split()
    rows = []
    for record in (stage / "ls-tree").read_bytes().split(b"\0"):
        if not record:
            continue
        meta, _, name = record.partition(b"\t")
        mode, kind, oid, size = meta.split()
        if kind != b"blob":
            raise RuntimeError("unexpected tree entry " + record.decode(errors="replace"))
        rows.append((int(mode, 8), oid.decode(), name.decode(), int(size)))
    write_index(rows, stage / "index")
    entries = read_container_git(stage / "git-raw.tar")
    if ".git/index" in entries or ".git" not in entries:
        raise RuntimeError("unexpected container .git layout")
    entries[".git/index"] = (0o644, (stage / "index").read_bytes())
    names = deterministic_tar(entries, work / "workspace-git.tar")
    # Native comparison: the SWE-smith mirror branch head (the native agent's checkout) tree.
    repository = source.mirror(repo_key, CACHE)
    native_tree = source.git(repository, "rev-parse", native_head + "^{tree}").decode().strip()
    native_rows = {}
    for record in source.git(repository, "ls-tree", "-r", "-z", native_head).split(b"\0"):
        if record:
            meta, _, name = record.partition(b"\t")
            mode, _, oid = meta.split()
            native_rows[name.decode()] = (int(mode, 8), oid.decode())
    ours = {name: (mode, oid) for mode, oid, name, _ in rows}
    differences = sorted(n for n in set(ours) | set(native_rows) if ours.get(n) != native_rows.get(n))
    expected_differences = sorted(m["path"] for m in materialized)
    tracked = sorted(ours)
    generated = sorted(set(file_manifest(package / "public/workspace")) - set(tracked))
    return {"commit": commit, "tree": tree, "branch": BRANCH, "identity": START_IDENTITY, "date": START_DATE,
            "message": START_MESSAGE, "git_version": (stage / "git-version").read_text().strip(),
            "archive": GIT_TAR, "archive_sha256": sha256(work / "workspace-git.tar"), "archive_entries": len(names),
            "tracked_files": len(tracked), "untracked_start_files": generated,
            "blobs": sorted({oid for _, oid, _, _ in rows}),
            "native_branch_head": native_head, "native_tree": native_tree, "native_tree_equal": native_tree == tree,
            "native_tree_differences": differences,
            "native_tree_differences_explained": differences == expected_differences,
            "native_tree_note": ("identical to the SWE-smith mirror branch head tree (the native agent's checkout)"
                                 if native_tree == tree else
                                 "differs only where documentation symlinks were replaced by copies (adaptation, as v1/v2)"),
            "start_mtime": START_MTIME, "git_mtime": GIT_MTIME, "index_rule": INDEX_RULE,
            "system_git_config": [f"{k}={v}" for k, v in SYSTEM_GIT_CONFIG]}


# --------------------------------------------------------------------------- worker-container git check
def _exec(env, command: str) -> tuple[int, str]:
    result = env.execute_recorded({"command": command}, timeout=300)
    if result.status in ("transport_error", "transport_timeout"):
        raise RuntimeError("worker command transport failed: " + command)
    return result.returncode, result.output


def git_start_check(adapter: SWESmithLongV3TaskAdapter, work: Path, *, forbidden_blobs: dict[str, str],
                    probe_file: str) -> dict:
    """Real worker container (DockerWorkerEnvironment, network none, runner's W transport).

    forbidden_blobs: git blob id -> label (withheld FAIL_TO_PASS files, clean versions of mutated files).
    """
    from trajectory_retention.execution.archives import pack_directory
    from trajectory_retention.execution.environment import DockerWorkerEnvironment, fidelity_manifest_from_tar
    workspace = work / "git-check" / "W"
    shutil.rmtree(workspace.parent, ignore_errors=True)
    adapter.materialize(workspace)
    git = adapter.git_start
    obs, a = {}, {}
    envs = []
    try:
        env = DockerWorkerEnvironment(image=adapter.manifest["environment_image_id"], network="none",
                                      name_prefix="tr-swesmith-long-v3-git")
        envs.append(env)
        data = pack_directory(workspace, default_owner=(env.worker_uid, env.worker_gid))
        env.restore_workspace(data, fidelity_manifest_from_tar(data))
        def run(label, command):
            code, out = _exec(env, command)
            obs[label] = {"returncode": code, "output": out[-4000:]}
            return code, out
        _, index_before = run("index_before", "sha256sum .git/index | cut -c1-64")
        code, status = run("git_status", "git status --porcelain=v1 --untracked-files=all")
        _, index_after = run("index_after", "sha256sum .git/index | cut -c1-64")
        a["git_status_clean_at_start"] = code == 0 and status.strip() == ""
        a["git_status_does_not_rewrite_index"] = index_before.strip() == index_after.strip() != ""
        _, log = run("git_log", "git log --all --format='%H %T|%an <%ae>|%cn <%ce>|%aI|%s'")
        a["git_log_exactly_one_commit"] = log.strip().splitlines() == [
            f"{git['commit']} {git['tree']}|{git['identity']['name']} <{git['identity']['email']}>|"
            f"{git['identity']['name']} <{git['identity']['email']}>|{git['date']}|{git['message']}"]
        _, count = run("rev_list_count", "git rev-list --all --count")
        a["rev_list_all_count_1"] = count.strip() == "1"
        _, refs = run("refs", "git for-each-ref --format='%(refname)'; git stash list | wc -l; "
                              "test -e .git/packed-refs && echo PACKED_REFS || true")
        a["only_branch_main_no_stash_tags_remotes"] = refs.split() == ["refs/heads/main", "0"]
        run("reflog", "git reflog --all --format='%H %gs'")
        code, fsck = run("fsck", "git fsck --full --strict --no-progress 2>&1; echo FSCK_RC=$?")
        a["fsck_clean_no_dangling"] = fsck.strip() == "FSCK_RC=0"
        _, objects = run("objects", "git cat-file --batch-all-objects --batch-check='%(objectname) %(objecttype)'")
        object_rows = [line.split() for line in objects.strip().splitlines()]
        kinds = Counter(kind for _, kind in object_rows)
        blobs = {oid for oid, kind in object_rows if kind == "blob"}
        obs["object_counts"] = dict(kinds)
        a["objects_exactly_start_commit"] = kinds["commit"] == 1 and kinds["tag"] == 0 and blobs == set(git["blobs"])
        leaks = sorted(label for oid, label in forbidden_blobs.items() if oid in {o for o, _ in object_rows})
        obs["forbidden_blob_hits"] = leaks
        a["no_withheld_or_reference_blob_in_git"] = not leaks
        withheld = [r["path"] for r in adapter.withheld["files"]]
        _, paths = run("history_paths", "git rev-list --objects --all | cut -d' ' -f2- | sort -u")
        a["no_withheld_path_in_history"] = not (set(paths.split("\n")) & set(withheld))
        _, config = run("config", "git config user.name; git config user.email; git config core.checkStat; "
                                  "git config core.trustctime")
        a["worker_identity_and_stat_config"] = config.strip().splitlines() == [
            WORKER_IDENTITY["name"], WORKER_IDENTITY["email"], "minimal", "false"]
        # Branchability: capture W as the runner does, restore into a fresh container.
        captured, live = env.capture_workspace()
        env2 = DockerWorkerEnvironment(image=adapter.manifest["environment_image_id"], network="none",
                                       name_prefix="tr-swesmith-long-v3-git")
        envs.append(env2)
        env2.restore_workspace(captured, live)
        code, restored = _exec(env2, "sha256sum .git/index | cut -c1-64; git status --porcelain=v1 --untracked-files=all; "
                                     "sha256sum .git/index | cut -c1-64; git rev-parse HEAD")
        obs["restored_status"] = {"returncode": code, "output": restored}
        lines = restored.strip().splitlines()
        a["restored_copy_clean_same_head_index_untouched"] = (code == 0 and lines == [index_before.strip()] * 2 + [git["commit"]])
        # Workflow the native agent has: diff, stash, stash pop, checkout.
        quoted = "'" + probe_file.replace("'", "'\\''") + "'"
        _, start_hash = run("probe_hash_start", f"sha256sum {quoted} | cut -c1-64")
        code, out = run("edit_and_diff", f"printf '\\n# git workflow probe\\n' >> {quoted} && git diff --name-only && git diff --stat | tail -1")
        a["git_diff_shows_edit"] = code == 0 and out.splitlines()[0] == probe_file
        code, out = run("stash", f"git stash && git status --porcelain=v1 && sha256sum {quoted} | cut -c1-64 && git stash list | wc -l")
        tail = out.strip().splitlines()
        a["git_stash_restores_start"] = code == 0 and tail[-2:] == [start_hash.strip(), "1"] and "M " not in out
        code, out = run("stash_pop", "git stash pop >/dev/null && git diff --name-only && git stash list | wc -l")
        a["git_stash_pop_restores_edit"] = code == 0 and out.strip().splitlines() == [probe_file, "0"]
        code, out = run("checkout", f"git checkout -q -- {quoted} && git status --porcelain=v1 --untracked-files=all | wc -l "
                                    f"&& sha256sum {quoted} | cut -c1-64")
        a["git_checkout_restores_start"] = code == 0 and out.strip().splitlines() == ["0", start_hash.strip()]
    finally:
        for env in envs:
            env.close()
    return {"image_id": adapter.manifest["environment_image_id"], "commit": git["commit"], "probe_file": probe_file,
            "assertions": a, "observations": obs, "passed": bool(a) and all(a.values())}


# --------------------------------------------------------------------------- leak scan (v2 rule + .git objects)
def git_objects(git_dir: Path) -> dict[str, bytes]:
    """Every loose object in git_dir, decompressed and verified (no packs are written at the start)."""
    if any((git_dir / "objects" / "pack").iterdir()):
        raise RuntimeError("unexpected pack files in the start repository")
    result = {}
    for path in sorted((git_dir / "objects").glob("[0-9a-f][0-9a-f]/*")):
        raw = zlib.decompress(path.read_bytes())
        oid = path.parent.name + path.name
        if hashlib.sha1(raw).hexdigest() != oid:
            raise RuntimeError("corrupt loose object " + oid)
        result[oid] = raw
    return result


def withheld_leak_scan(package: Path, natural: Path, withheld: dict[str, bytes], needles: list[str], key: str) -> dict:
    """v2's rule (pipeline_v2.withheld_leak_scan) against the v3 image, plus every decompressed .git object."""
    hashes = {hashlib.sha256(d).hexdigest(): p for p, d in withheld.items()}
    files = [f for f in natural.rglob("*") if f.is_file() and not f.is_symlink()]
    present = [p for p in withheld if (natural / p).exists() or (natural / p).is_symlink()]
    copies = [{"path": f.relative_to(natural).as_posix(), "withheld": hashes[sha256(f)]} for f in files if sha256(f) in hashes]
    wanted = set(needles)
    line_hits = []
    contents = [(f.relative_to(natural).as_posix(), f.read_bytes()) for f in files]
    objects = git_objects(natural / ".git")
    contents += [(".git object " + oid, raw.split(b"\0", 1)[1]) for oid, raw in objects.items()]
    object_copies = [{"object": oid, "withheld": hashes[hashlib.sha256(raw.split(b"\0", 1)[1]).hexdigest()]}
                     for oid, raw in objects.items() if hashlib.sha256(raw.split(b"\0", 1)[1]).hexdigest() in hashes]
    for name, data in contents:
        found = {line.strip() for line in data.decode("utf-8", "ignore").splitlines()} & wanted
        if found:
            line_hits.append({"path": name, "lines": sorted(found)[:3]})
    inventory = json.loads((EVIDENCE_V3 / "families" / key / "image-inventory.json").read_text())
    image_copies = [{"image_path": path, "withheld": hashes[d]} for path, _, d in inventory if d in hashes]
    image_needles = json.loads((EVIDENCE_V3 / "families" / key / "image-needles.json").read_text())
    per_file = json.loads((EVIDENCE_V3 / "families" / key / "needles.json").read_text())
    v2_name = package.name[:-3] + "-v2"
    task_hits, image_line_hits = [], []
    for hit in image_needles["hits"]:
        task, _, path = hit["owner"].partition("::")
        if task != v2_name:
            continue
        total = len(per_file[hit["owner"]])
        threshold = total if total < 3 else max(3, -(-total // 5))
        hit = {**hit, "withheld_file": path, "file_needles": total, "threshold": threshold}
        task_hits.append(hit)
        if hit["count"] >= threshold:
            image_line_hits.append(hit)
    return {"withheld_paths_present_in_W": present, "identical_copies_in_W": copies,
            "git_objects_scanned": len(objects), "identical_copies_in_git_objects": object_copies,
            "distinctive_lines_checked": len(needles), "distinctive_line_hits_in_W": line_hits,
            "identical_copies_in_image": image_copies, "distinctive_line_hits_in_image": image_line_hits,
            "image_matches_below_threshold": sorted((h for h in task_hits if h["count"] < h["threshold"]),
                                                    key=lambda h: -h["count"])[:10],
            "image_max_fraction_of_a_withheld_file_matched": max((h["count"] / h["file_needles"] for h in task_hits), default=0.0),
            "image_files_scanned": image_needles["files_scanned"],
            "leak": bool(present or copies or object_copies or line_hits or image_copies or image_line_hits),
            "rule": "v2 rule on the v3 image (withheld paths absent from W; no W or image file byte-identical to a "
                    "withheld file; no W file containing a distinctive line of a withheld file; no image file with "
                    ">= max(3, 20%) of one withheld file's distinctive lines), extended to every decompressed .git "
                    "object (no object equal to, or containing a distinctive line of, a withheld file)"}


# --------------------------------------------------------------------------- validation (v2 gates)
def validate(package: Path, work: Path, row: dict, out: Path) -> dict:
    """The v2 gates (pipeline.validate) on the v3 start, with the three .git-dependent checks restated.

    Changed checks: natural start = package workspace + frozen .git (was: package workspace);
    Git start verified in a worker container (was: no .git); reference-copy scan over W files and
    .git objects (was: files, and no .git). Graded trees keep .git, so every grade also exercises
    the grader's .git skipping.
    """
    adapter = SWESmithLongV3TaskAdapter(package, scratch_root=CACHE / "grade-scratch")
    natural = work / "natural"
    adapter.materialize(natural)
    no_git = {k: v for k, v in file_manifest(natural).items() if not (k == ".git" or k.startswith(".git/"))}
    extracted = {k[len(".git/"):]: v for k, v in file_manifest(natural).items() if k.startswith(".git/")}
    with tarfile.open(package / GIT_TAR) as archive:
        frozen = {m.name[len(".git/"):]: hashlib.sha256(archive.extractfile(m).read()).hexdigest()
                  for m in archive.getmembers() if m.isfile()}
    natural_equal = no_git == file_manifest(package / "public/workspace") and extracted == frozen
    mtimes_fixed = all(int(p.lstat().st_mtime) == START_MTIME for p in natural.rglob("*")
                       if ".git" not in p.relative_to(natural).parts)
    grades = {}
    for label in ("start-1", "start-2"):
        grades[label] = pipeline._grade(adapter, natural, out, label)
    reference = work / "reference"
    source.copy_tree(natural, reference)
    subprocess.run(["git", "apply", "-R", "-"], cwd=reference, input=row["patch"].encode(), check=True,
                   capture_output=True, env=source._env(reference))
    for label in ("reference-1", "reference-2"):
        grades[label] = pipeline._grade(adapter, reference, out, label)
    partial, chosen = [], None
    hunks = pipeline._hunk_patches(row["patch"])
    attempts = [(name, [hunk]) for name, hunk in hunks]
    if len(hunks) > 2:
        attempts += [("all-but-" + name, [h for n, h in hunks if n != name]) for name, _ in hunks]
    for name, selected in attempts:
        label = "partial-" + re.sub(r"[^A-Za-z0-9]+", "-", name)
        tree = work / label
        source.copy_tree(natural, tree)
        applied = all(subprocess.run(["git", "apply", "-R", "-"], cwd=tree, input=h.encode(), capture_output=True,
                                     env=source._env(tree)).returncode == 0 for h in selected)
        if not applied:
            partial.append({"hunk": name, "applied": False})
            continue
        result = pipeline._grade(adapter, tree, out, label)
        partial.append({"hunk": name, "applied": True, **{k: v for k, v in result.items() if k != "_statuses"}})
        if result["valid"] and 0 < result["score"] < 1:
            chosen = name
            break
    tamper = work / "tamper"
    source.copy_tree(natural, tamper)
    for rel in adapter.grading["protected_dirs"]:
        shutil.rmtree(tamper / rel, ignore_errors=True)
    protected = set(adapter.grading["protected_files"])
    for rel in {x.split("::", 1)[0] for x in adapter.scorer.f2p} & protected:
        path = tamper / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_tampered():\n    assert True\n")
    (tamper / "conftest.py").write_text("import pytest\n\n@pytest.hookimpl(tryfirst=True)\ndef pytest_runtest_call(item):\n    pass\n")
    grades["tamper-tests"] = pipeline._grade(adapter, tamper, out, "tamper-tests")
    starts = [grades["start-1"], grades["start-2"]]
    refs = [grades["reference-1"], grades["reference-2"]]
    leak = pipeline.start_leak_scan(natural, row["patch"], work)
    reference_blobs = {git_blob_id((work / "clean-mutated" / f["a"]).read_bytes()): "reference:" + f["a"]
                       for f in pipeline.patch_files(row["patch"])}
    withheld_blobs = {git_blob_id((package / "private/pristine" / r["path"]).read_bytes()): "withheld:" + r["path"]
                      for r in adapter.withheld["files"]}
    probe_file = adapter.manifest["bug_shape"]["files"][0]
    git_check = git_start_check(adapter, work, forbidden_blobs={**reference_blobs, **withheld_blobs}, probe_file=probe_file)
    write_json(out / "git-start-check.json", git_check)
    skipped = {label: _skipped(out / label) for label in grades}
    checks = {
        "natural_start_equals_package_plus_frozen_git": natural_equal,
        "start_files_have_fixed_mtime": mtimes_fixed,
        "git_start_verified_in_worker_container": git_check["passed"],
        "grader_skipped_only_git_metadata": all(v == [".git"] for v in skipped.values()),
        "start_valid": all(g["valid"] for g in starts),
        "start_no_f2p_passing": all(g["f2p_passed"] == 0 for g in starts),
        "start_deterministic": starts[0]["statuses_sha256"] == starts[1]["statuses_sha256"],
        "reference_valid": all(g["valid"] for g in refs),
        "reference_full_solve": all(g["full_solve"] is True and g["score"] == 1.0 for g in refs),
        "reference_deterministic": refs[0]["statuses_sha256"] == refs[1]["statuses_sha256"],
        "partial_control_strictly_between": chosen is not None,
        "tamper_control_equals_start": grades["tamper-tests"]["statuses_sha256"] == starts[0]["statuses_sha256"],
        "start_has_no_reference_copy": (not leak["reference_file_copies"] and leak["git_metadata"] == [".git"]
                                        and "forbidden_blob_hits" in git_check["observations"]
                                        and not any(h.startswith("reference:")
                                                    for h in git_check["observations"]["forbidden_blob_hits"])),
    }
    failures = [k for k, v in checks.items() if not v]
    times = [g["wall_seconds"] for g in grades.values() if g.get("wall_seconds") is not None] + [
        p["wall_seconds"] for p in partial if p.get("wall_seconds") is not None]
    start_statuses = starts[0]["_statuses"]
    summary = {"passed": not failures, "failures": failures, "checks": checks,
               "grades": {k: {x: y for x, y in v.items() if x != "_statuses"} for k, v in grades.items()},
               "grader_vcs_metadata_skipped": skipped,
               "partial_controls": partial, "partial_control_hunk": chosen, "start_leak_scan": leak,
               "start_f2p_status_counts": dict(Counter(start_statuses[c] for c in adapter.scorer.f2p)),
               "start_p2p_status_counts": dict(Counter(start_statuses[c] for c in adapter.scorer.p2p)),
               "grade_seconds": {"max": max(times), "median": sorted(times)[len(times) // 2]},
               "grade_time_flag_over_120s": max(times) > 120,
               "evaluator_version": adapter.scorer.version, "git_start_check": git_check}
    write_json(out / "validation.json", summary)
    return {k: v for k, v in summary.items() if k not in ("grades", "partial_controls", "git_start_check")} | {
        "grades": {k: {x: y for x, y in v.items() if x in ("valid", "score", "full_solve", "wall_seconds", "status_counts",
                                                           "statuses_sha256")}
                   for k, v in summary["grades"].items()},
        "partial_controls": [{k: p.get(k) for k in ("hunk", "applied", "score", "valid", "statuses_sha256")} for p in partial],
        "git_start_check": {k: git_check[k] for k in ("commit", "probe_file", "assertions", "passed")}
                           | {"object_counts": git_check["observations"].get("object_counts")}}


def _skipped(grade_dir: Path) -> list[str] | None:
    driver = grade_dir / "driver.json"
    return json.loads(driver.read_text()).get("vcs_metadata_skipped") if driver.is_file() else None


def compare_with_v2(v3_validation: dict, v2_id: str) -> dict:
    """Grades of the v3 trees vs the v2 raw evidence of the same trees (per-check statuses hash, score)."""
    v2 = json.loads((EVIDENCE_V2 / "tasks" / v2_id / "validation.json").read_text())
    rows, equal = {}, True
    for label in ("start-1", "start-2", "reference-1", "reference-2", "tamper-tests"):
        a, b = v2["grades"][label], v3_validation["grades"][label]
        same = (a["statuses_sha256"], a["score"], a["valid"], a["full_solve"]) == (
            b["statuses_sha256"], b["score"], b["valid"], b["full_solve"])
        rows[label] = {"v2_statuses_sha256": a["statuses_sha256"], "v3_statuses_sha256": b["statuses_sha256"],
                       "v2_score": a["score"], "v3_score": b["score"], "equal": same}
        equal &= same
    v2_partial = {p["hunk"]: p for p in v2["partial_controls"] if p.get("applied")}
    partial_rows = []
    for p in v3_validation["partial_controls"]:
        if not p.get("applied"):
            continue
        old = v2_partial.get(p["hunk"])
        same = old is not None and (old["statuses_sha256"], old["score"]) == (p["statuses_sha256"], p["score"])
        partial_rows.append({"hunk": p["hunk"], "v2_score": old and old["score"], "v3_score": p["score"], "equal": same})
        equal &= same
    same_hunk = v2["partial_control_hunk"] == v3_validation["partial_control_hunk"]
    return {"v2_evidence": f"artifacts/task-admission/swesmith-long/v2/tasks/{v2_id}/validation.json",
            "grades": rows, "partial_controls": partial_rows, "partial_control_hunk_equal": same_hunk,
            "all_equal": bool(equal and same_hunk)}


# --------------------------------------------------------------------------- package build
def build(v2_id: str, family: dict) -> dict:
    task_id = v3_id(v2_id)
    out = EVIDENCE_V3 / "tasks" / task_id
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    work = WORK / task_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    record = {"task_id": task_id, "predecessor": v2_id, "status": "running"}
    try:
        if v2_id in HOLD_V2:
            record.update(status="HOLD", reason=HOLD_V2[v2_id])
            return record
        old = PACKAGES_V2 / v2_id
        old_manifest = json.loads((old / "package.json").read_text())
        row = json.loads((old / "private/source-row.json").read_text())
        record.update(instance_id=row["instance_id"], repo_key=old_manifest["repo_key"],
                      family_id=old_manifest["family_id"], split=old_manifest["planned_split"])
        from .adapter_v2 import SWESmithLongV2TaskAdapter
        SWESmithLongV2TaskAdapter(old, scratch_root=CACHE / "grade-scratch")  # v2 package still verifies
        package = PACKAGES / task_id
        shutil.rmtree(package, ignore_errors=True)
        package.mkdir(parents=True)
        shutil.copytree(old / "private", package / "private", symlinks=True)
        (package / "public").mkdir()
        source.copy_tree(old / "public/workspace", package / "public/workspace")
        shutil.copy2(old / "public/instructions.md", package / "public/instructions.md")
        write_json(package / "private/adaptation.json", ADAPTATION)
        materialized = (old_manifest["adaptation"].get("symlinks_materialized")
                        or old_manifest["mirror"].get("symlinks_materialized") or [])
        git_start = build_git_start(package, family["image_id"], work, old_manifest["native_start"]["mirror_branch_head"],
                                    old_manifest["repo_key"], materialized)
        shutil.copy2(work / "workspace-git.tar", package / GIT_TAR)
        record["git_start"] = {k: v for k, v in git_start.items() if k != "blobs"}
        if not git_start["native_tree_differences_explained"]:
            raise RuntimeError("start commit tree differs from the native branch head beyond disclosed symlink copies: "
                               + str(git_start["native_tree_differences"][:5]))
        from ..prompts import SWESMITH_INSTANCE_TEMPLATE_SHA256, swesmith_prompt_v3
        from trajectory_retention.contracts import StageSpec
        text = (package / "public/instructions.md").read_text()
        prompt = swesmith_prompt_v3(StageSpec(stage_id="repair", target_id=task_id + ":repair", ordinal=1,
                                              instructions=text, requirement_version="", evaluator_version=""))
        manifest = {**{k: v for k, v in old_manifest.items() if k not in ("public_files", "private_files")},
                    "schema_version": SCHEMA, "task_id": task_id, "package_version": task_id,
                    "adapter_entry": ADAPTER_ENTRY, "adaptation": {**ADAPTATION, "symlinks_materialized": materialized},
                    "evaluator_sources": evaluator_sources(),
                    "predecessor": {"task_id": v2_id, "package_sha256": sha256(old / "package.json")},
                    "environment_image_id": family["image_id"], "environment_tag": family["tag"],
                    "environment_git": {k: family[k] for k in ("base_image_id", "base_image_tag", "dockerfile_sha256",
                                                               "git_version", "system_git_config", "dpkg_added")},
                    "git_start": git_start,
                    "worker_prompt": {"version": prompt.version, "rendered_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
                                      "template_sha256": SWESMITH_INSTANCE_TEMPLATE_SHA256, "source": prompt.source},
                    "public_files": file_manifest(package / "public"), "private_files": file_manifest(package / "private")}
        write_json(package / "package.json", manifest)
        record["package"] = str(package.relative_to(WORKTREE))
        record["package_sha256"] = sha256(package / "package.json")
        record["counts"] = old_manifest["counts"]
        record["bug_shape"] = old_manifest["bug_shape"]
        record["native_start"] = old_manifest["native_start"]
        record["validation"] = validate(package, work, row, out)
        withheld = json.loads((package / "private/withheld.json").read_text())["files"]
        withheld_bytes = {r["path"]: (package / "private/pristine" / r["path"]).read_bytes() for r in withheld}
        per_file = json.loads((EVIDENCE_V3 / "families" / record["repo_key"] / "needles.json").read_text())
        needles = sorted({line for r in withheld for line in per_file[v2_id + "::" + r["path"]]})
        leak = withheld_leak_scan(package, work / "natural", withheld_bytes, needles, record["repo_key"])
        record["withheld_leak_scan"] = leak
        record["v2_comparison"] = compare_with_v2(record["validation"], v2_id)
        checks = record["validation"]["checks"]
        checks["withheld_tests_absent_from_workspace_git_and_image"] = not leak["leak"]
        checks["grades_equal_v2"] = record["v2_comparison"]["all_equal"]
        record["validation"]["failures"] = [k for k, v in checks.items() if not v]
        record["validation"]["passed"] = not record["validation"]["failures"]
        record["status"] = "VALIDATED" if record["validation"]["passed"] else "HOLD"
        if record["status"] == "HOLD":
            record["reason"] = "; ".join(record["validation"]["failures"])
        return record
    except Exception as exc:
        record.update(status="HOLD", reason=f"{type(exc).__name__}: {exc}"[:2000])
        return record
    finally:
        write_json(out / "package-record.json", record)
