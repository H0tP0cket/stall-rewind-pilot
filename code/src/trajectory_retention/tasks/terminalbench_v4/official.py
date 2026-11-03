"""Identify the exact base release each official TB4 image was built from (registry metadata only; nothing is pulled).

Adapted from the terminalbench-v2 lane's ``official.py`` (lanes never import each other). The v4.0.0 release publishes
``prebuilt-images.json``: per task an environment image and a verifier image in ``harborframework/terminal-bench``,
linux/amd64 only. Their config blobs (a few KB) record the base's root-filesystem layer diff IDs and, for python bases,
``PYTHON_VERSION``. A candidate base tag matches when its linux/amd64 diff IDs are a prefix of the official image's; the
same tag's linux/arm64 variant is then pinned by digest for the native rebuild, so the rebuilt environment starts from
the same upstream base release (for example CPython 3.12.14 rather than whatever ``python:3.12-slim-bookworm`` is today).
A Dockerfile that pins an amd64-only digest (``ubuntu:24.04@sha256:...``) is matched the same way through the official
image's layers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

from . import source as S

CACHE = Path.home() / ".cache/terminalbench/registry-cache"
RELEASE_MANIFEST_URL = f"{S.SOURCE_URL}/releases/download/{S.SOURCE_TAG}/prebuilt-images.json"
INDEX_TYPES = ("application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,"
               "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json")
SERIES = {"24.04": "noble", "22.04": "jammy", "noble": "noble", "jammy": "jammy"}


def _get(url: str, token: str | None = None, accept: str | None = None) -> bytes:
    """GET with an on-disk cache for immutable or slow-changing metadata (Docker Hub rate-limits)."""
    cacheable = ("/blobs/sha256:" in url or "/manifests/" in url or "hub.docker.com" in url
                 or url == RELEASE_MANIFEST_URL)
    key = CACHE / hashlib.sha256((url + "|" + (accept or "")).encode()).hexdigest()
    if cacheable and "auth.docker.io" not in url and key.is_file():
        return key.read_bytes()
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    if accept:
        request.add_header("Accept", accept)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if cacheable and "auth.docker.io" not in url:
        CACHE.mkdir(parents=True, exist_ok=True)
        key.write_bytes(data)
    return data


def release_manifest() -> dict:
    data = json.loads(_get(RELEASE_MANIFEST_URL))
    if data.get("release_tag") != S.SOURCE_TAG:
        raise ValueError("Release manifest is not v4.0.0")
    return data


def official_images(task: str) -> dict[str, dict]:
    """{role: image row} for one task from the release manifest (roles: environment, verifier)."""
    rows = [t for t in release_manifest()["tasks"] if t.get("task") == task]
    if len(rows) != 1:
        raise LookupError(f"{task}: not exactly one release manifest entry")
    if rows[0].get("source_sha") != S.SOURCE_REVISION:
        raise ValueError(f"{task}: release images were not built from the pinned revision")
    return {img["role"]: img for img in rows[0]["images"]}


def token(repo: str) -> str:
    return json.loads(_get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"))["token"]


def manifest(repo: str, ref: str, tok: str) -> dict:
    return json.loads(_get(f"https://registry-1.docker.io/v2/{repo}/manifests/{ref}", tok, INDEX_TYPES))


def config(repo: str, ref: str, *, arch: str = "amd64") -> tuple[dict, str]:
    """(config blob, platform manifest digest) of ``repo:ref`` for linux/<arch>."""
    tok = token(repo)
    m = manifest(repo, ref, tok)
    digest = ref if ref.startswith("sha256:") else ""
    if "manifests" in m:
        pick = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == arch
                and x.get("platform", {}).get("os") == "linux"]
        if not pick:
            raise LookupError(f"{repo}:{ref} has no linux/{arch} variant")
        digest = pick[0]["digest"]
        m = manifest(repo, digest, tok)
    blob = json.loads(_get(f"https://registry-1.docker.io/v2/{repo}/blobs/{m['config']['digest']}", tok))
    if blob.get("architecture") != arch:
        raise LookupError(f"{repo}@{ref} is {blob.get('architecture')}, not {arch}")
    return blob, digest


def official_config(task: str, role: str) -> tuple[dict, dict]:
    row = official_images(task)[role]
    repo, digest = row["pinned_ref"].split(":", 1)[0], row["digest"]
    blob, _ = config(repo, digest)
    return blob, row


def hub_tags(repo: str, prefix: str) -> list[str]:
    out, url = [], f"https://hub.docker.com/v2/repositories/{repo}/tags?name={prefix}&page_size=100"
    while url:
        data = json.loads(_get(url))
        out += [r["name"] for r in data.get("results", [])]
        url = data.get("next")
    return out


def split_ref(ref: str) -> tuple[str, str, str | None]:
    """(name, tag, digest) of a FROM reference such as ``ubuntu:24.04@sha256:...``."""
    ref, _, digest = ref.partition("@")
    name, _, tag = ref.partition(":")
    return name, tag or "latest", digest or None


def candidate_tags(ref: str, official: dict) -> list[str]:
    name, tag, _ = split_ref(ref)
    env = dict(e.split("=", 1) for e in official.get("config", {}).get("Env", []) if "=" in e)
    if name == "python" and "PYTHON_VERSION" in env:
        minor = ".".join(env["PYTHON_VERSION"].split(".")[:2])
        return [tag.replace(minor, env["PYTHON_VERSION"], 1)] if minor in tag else []
    if name == "ubuntu" and tag in SERIES:
        series = SERIES[tag]
        base_created = next((h.get("created", "") for h in official.get("history", []) if "ADD file:" in
                             h.get("created_by", "")), "")
        limit = re.sub(r"[^0-9]", "", base_created[:10]) or "99999999"
        tags = [t for t in hub_tags("library/ubuntu", series + "-20") if re.fullmatch(series + r"-\d{8}(\.\d+)?", t)]
        # Newest dated tag not after the official base layer's creation date first.
        return sorted((t for t in tags if t[len(series) + 1:len(series) + 9] <= limit), reverse=True)
    return [tag]


def match_base(ref: str, official: dict) -> dict:
    """Find the base tag whose amd64 layers prefix the official image; return its arm64 digest pin."""
    name, tag, digest = split_ref(ref)
    repo = name if "/" in name else "library/" + name
    wanted = official["rootfs"]["diff_ids"]
    tried = []
    if digest:
        try:  # the Dockerfile's own pin: record its platform and whether it prefixes the official image
            pinned, _ = config(repo, digest)
            tried.append({"tag": f"{tag}@{digest}", "layers": len(pinned["rootfs"]["diff_ids"]),
                          "prefix_match": wanted[:len(pinned["rootfs"]["diff_ids"])] == pinned["rootfs"]["diff_ids"],
                          "note": "the Dockerfile's digest pin (an amd64 manifest, not an index)"})
        except Exception as exc:  # noqa: BLE001 - recorded
            tried.append({"tag": f"{tag}@{digest}", "error": str(exc)[:200]})
    for cand in candidate_tags(ref, official):
        try:
            amd, _ = config(repo, cand, arch="amd64")
        except Exception as exc:  # noqa: BLE001 - recorded
            tried.append({"tag": cand, "error": str(exc)[:200]})
            continue
        ids = amd["rootfs"]["diff_ids"]
        ok = wanted[:len(ids)] == ids
        tried.append({"tag": cand, "layers": len(ids), "prefix_match": ok})
        if ok:
            arm, arm_digest = config(repo, cand, arch="arm64")
            return {"ref": ref, "match": "layer_prefix", "matched_tag": f"{name}:{cand}",
                    "pinned": f"{name}:{cand}@{arm_digest}", "amd64_layers_matched": len(ids),
                    "arm64_created": arm.get("created"),
                    "arm64_env": [e for e in arm.get("config", {}).get("Env", []) if "VERSION" in e],
                    "candidates_tried": tried}
    env = dict(e.split("=", 1) for e in official.get("config", {}).get("Env", []) if "=" in e)
    if name == "python" and "PYTHON_VERSION" in env and tried and "error" not in tried[-1]:
        # Layer digests of re-published python tags do not reproduce; match the CPython release instead.
        cand = tried[-1]["tag"]
        arm, arm_digest = config(repo, cand, arch="arm64")
        return {"ref": ref, "match": "python_version_only", "matched_tag": f"{name}:{cand}",
                "pinned": f"{name}:{cand}@{arm_digest}", "official_python_version": env["PYTHON_VERSION"],
                "arm64_created": arm.get("created"),
                "arm64_env": [e for e in arm.get("config", {}).get("Env", []) if "VERSION" in e],
                "candidates_tried": tried}
    return {"ref": ref, "match": None, "matched_tag": None, "candidates_tried": tried}


def pull_arm64(pinned: str) -> str:
    present = subprocess.run(["docker", "image", "inspect", pinned], capture_output=True)
    if present.returncode == 0:
        return pinned
    subprocess.run(["docker", "pull", "--platform", "linux/arm64", pinned], check=True, capture_output=True, timeout=1800)
    return pinned
