"""Pinned MettleBench source identity, task families and the frozen family split.

Upstream: https://github.com/Kelvin-Coffee/MettleBench (task prompts, ordered
checklists, evaluators) and the Hugging Face dataset KelvinCoffee/MettleBench
(per-task workspace archives). The gated reference-run dataset was not
available; admitted tasks use independently authored private witnesses.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess

SOURCE_URL = "https://github.com/Kelvin-Coffee/MettleBench"
SOURCE_REVISION = "bdfc9730839544c34082b208552d2b93da8be81b"
DATASET_URL = "https://huggingface.co/datasets/KelvinCoffee/MettleBench"
DATASET_REVISION = "e75c6142261fbfab2396849b5935383e74e70e2a"
PAPER = "AgentRewind: Recoverable Execution for Long-Horizon LLM Agents, arXiv:2608.14380"
DEFAULT_CACHE = Path.home() / ".cache/mettlebench"

# Native Continue protocol (paper section "Experiments", appendix B.1/B.3):
# every submission runs the ordered checklist; only the first unsatisfied
# criterion's feedback string is returned as an ordinary user message; a run
# ends on success or when five consecutive submissions fail at the same
# criterion. Score = satisfied checklist prefix / active criteria.
NATIVE_PROTOCOL = {
    "version": "mettlebench.native-continue.v1",
    "submission_feedback": "first_failure.feedback only, appended verbatim as an ordinary user message",
    "success": "all active criteria pass",
    "repeated_failure_termination": 5,
    "repeated_failure_rule": "same first unsatisfied criterion on five consecutive submissions",
    "native_score": "passed prefix / active criteria (upstream _verifier_v2.legacy_result_to_v2 pass_rate)",
    "source_evidence": [
        f"{SOURCE_URL}/blob/{SOURCE_REVISION}/README.md",
        f"{SOURCE_URL}/blob/{SOURCE_REVISION}/tasks/_verifier_v2.py",
        PAPER + " (Experiments; Appendix A.5, B.1 Table 9, B.3 Continue)",
    ],
    "unpublished_detail": "The exact user-message wrapper of the authors' batch driver is not published; the bare feedback string is delivered.",
}

# Terminal-Bench 2.0 task each MettleBench terminal_* task was derived from
# (task premises/environments per NOTICE.md; names from the pinned TB2 census).
TERMINAL_SOURCES = {
    "terminal_caffe_cifar_snapshot_handoff": "caffe-cifar-10",
    "terminal_cobol_migration_audit": "cobol-modernization",
    "terminal_code_image_handoff": "code-from-image",
    "terminal_cython_numpy_route_handoff": "build-cython-ext",
    "terminal_cython_pyx_regen_handoff": "build-cython-ext",
    "terminal_db_wal_recovery_handoff": "db-wal-recovery",
    "terminal_dna_assembly_handoff": "dna-assembly",
    "terminal_fasttext_seed_model_handoff": "train-fasttext",
    "terminal_financial_document_handoff": "financial-document-processor",
    "terminal_git_fix_dangling_handoff": "fix-git",
    "terminal_git_leak_graveyard_handoff": "git-leak-recovery",
    "terminal_git_multibranch_topology_lockin": "git-multibranch",
    "terminal_git_sanitize_filter_repo_handoff": "sanitize-git-repo",
    "terminal_grpc_protoc_regen_handoff": "kv-store-grpc",
    "terminal_hf_sentiment_cache_handoff": "hf-model-inference",
    "terminal_mcmc_stan_data_handoff": "mcmc-sampling-stan",
    "terminal_multi_source_data_handoff": "multi-source-data-merger",
    "terminal_openssl_ca_resign_handoff": "openssl-selfsigned-cert",
    "terminal_openssl_key_rotate_handoff": "openssl-selfsigned-cert",
    "terminal_overfull_editorial_handoff": "overfull-hbox",
    "terminal_pmars_source_handoff": "build-pmars",
    "terminal_povray_22_route_handoff": "build-pov-ray",
    "terminal_protein_assembly_handoff": "protein-assembly",
    "terminal_pypi_release_handoff": "pypi-server",
    "terminal_pytorch_checkpoint_handoff": "pytorch-model-recovery",
    "terminal_rstan_migration_handoff": "rstan-to-pystan",
    "terminal_sam_mask_handoff": "sam-cell-seg",
    "terminal_write_compressor_handoff": "write-compressor",
}

# Source project for non-terminal tasks. Tasks built on the same project share
# a family (cross-benchmark copies keep the repository's existing family id).
PROJECT_FAMILIES = {
    "gittaskbench_eparse": "project:ChrisPappalardo/eparse",
    "gittaskbench_pypdf": "project:py-pdf/pypdf",
    "gittaskbench_stegano": "project:cedricbonhomme/Stegano",
    "gittaskbench_watermark": "project:ShieldMnt/invisible-watermark",
    "programbench_age": "project:FiloSottile/age",
    "programbench_atlas": "project:ariga/atlas",
    "programbench_bat": "project:sharkdp/bat",
    "programbench_calcurse": "project:lfos/calcurse",
    "programbench_chamber": "project:segmentio/chamber",
    "programbench_chroma": "project:alecthomas/chroma",
    "programbench_ctags": "project:universal-ctags/ctags",
    "programbench_gdu": "project:dundee/gdu",
    "programbench_git_trim": "project:foriequal0/git-trim",
    "programbench_gomplate": "project:hairyhenderson/gomplate",
    "programbench_gowsdl": "project:hooklift/gowsdl",
    "programbench_hostctl": "project:guumaster/hostctl",
    "programbench_jsonschema": "programbench:jsonschema",
    "programbench_lazygit": "project:jesseduffield/lazygit",
    "programbench_pandoc": "project:jgm/pandoc",
    "programbench_ripgrep": "project:BurntSushi/ripgrep",
    "programbench_samtools": "project:samtools/samtools",
    "programbench_skeema": "project:skeema/skeema",
    "programbench_svd2rust": "project:rust-embedded/svd2rust",
    "programbench_trdsql": "project:noborus/trdsql",
    "programbench_zoxide": "project:ajeetdsouza/zoxide",
    "projecteval_budget": "projecteval:budget",
    "projecteval_crm": "projecteval:crm",
    "projecteval_expense": "projecteval:expense",
    "projecteval_summary_workbook": "projecteval:summary-workbook",
    "swe_astropy": "project:astropy/astropy",
    "swe_cryptography": "project:pyca/cryptography",
    "swe_django": "project:django/django",
    "swe_matplotlib": "project:matplotlib/matplotlib",
    "swe_numpy": "project:numpy/numpy",
    "swe_pydicom": "project:pydicom/pydicom",
    "swe_scikit": "project:scikit-learn/scikit-learn",
    "swe_sphinx": "project:sphinx-doc/sphinx",
    # Existing repository family identities (already development/calibration).
    "swe_sqlfluff": "github:sqlfluff/sqlfluff",
    "swe_xarray": "project:pydata/xarray",
}

# Families already used for development/calibration elsewhere in the registry
# can never be held out (task-admission-criteria.md, "Family split").
FORCED_NON_HELD_OUT = {"github:sqlfluff/sqlfluff": "existing SWE-smith sqlfluff calibration family",
                       "project:pydata/xarray": "existing mettle-minixr development family"}
SPLIT_SEED = 20260923
SPLIT_RULE = ("mettlebench.family-split.v1: sort all family ids; forced families are calibration; "
              "shuffle the remaining families with random.Random(20260923); the first floor(n/2) are held_out, "
              "the rest calibration. Fixed before any model run on these tasks.")


def family_for(task_id: str) -> str:
    if task_id in TERMINAL_SOURCES:
        return "terminal-bench:" + TERMINAL_SOURCES[task_id]
    for prefix, family in sorted(PROJECT_FAMILIES.items(), key=lambda kv: -len(kv[0])):
        if task_id.startswith(prefix + "_"):
            return family
    raise KeyError("No declared family for " + task_id)


def family_split(task_ids) -> dict[str, str]:
    families = sorted({family_for(t) for t in task_ids})
    free = [f for f in families if f not in FORCED_NON_HELD_OUT]
    random.Random(SPLIT_SEED).shuffle(free)
    held = set(free[: len(free) // 2])
    return {f: ("held_out" if f in held else "calibration") for f in families}


def package_task_id(upstream_task_id: str) -> str:
    return "mettlebench-" + upstream_task_id.replace("_", "-") + "-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source(cache: Path = DEFAULT_CACHE) -> dict:
    """Check the pinned git checkout and every archive against HF LFS metadata."""
    src, hf = cache / "src", cache / "hf"
    head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_REVISION:
        raise ValueError("MettleBench checkout differs from the pinned revision")
    dirty = subprocess.check_output(["git", "-C", str(src), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("MettleBench checkout has modified tracked files")
    manifest = [json.loads(line) for line in (src / "manifest.jsonl").read_text().splitlines() if line.strip()]
    lfs = {e["path"].split("/")[-1][:-len(".tar.gz")]: e["lfs"] for e in json.loads((hf / "lfs-tree.json").read_text())}
    archives = {}
    for row in manifest:
        tid = row["task_id"]
        path = hf / "workspaces" / f"{tid}.tar.gz"
        digest = sha256_file(path)
        if digest != lfs[tid]["oid"] or path.stat().st_size != lfs[tid]["size"]:
            raise ValueError("Workspace archive differs from dataset LFS metadata: " + tid)
        archives[tid] = {"sha256": digest, "size": path.stat().st_size}
    return {"source_revision": head, "dataset_revision": DATASET_REVISION, "tasks": len(manifest),
            "manifest": manifest, "archives": archives}
