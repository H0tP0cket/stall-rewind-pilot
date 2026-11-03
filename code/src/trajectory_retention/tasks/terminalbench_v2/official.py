"""(Copied unchanged from the terminalbench-v1 lane; lanes never import each other.)

Identify the exact base image the official TB2 task image was built from (registry metadata only).

The official ``alexgshaw/<task>:20251031`` images are x86_64-only and are never
pulled. Their config blob (a few KB) records the base's root-filesystem layer
diff IDs and, for python bases, ``PYTHON_VERSION``. A candidate base tag matches
when its linux/amd64 config's diff IDs are a prefix of the official image's diff
IDs; the same tag's linux/arm64 variant is then used for the native rebuild, so
the rebuilt environment starts from the same upstream base release (e.g.
CPython 3.13.7 rather than whatever ``python:3.13-slim-bookworm`` means today).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import urllib.request

CACHE = Path.home() / ".cache/terminalbench/registry-cache"

INDEX_TYPES = ("application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,"
               "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json")


def _get(url: str, token: str | None = None, accept: str | None = None) -> bytes:
    """GET with an on-disk cache for immutable/slow-changing registry metadata (Docker Hub rate-limits pulls)."""
    cacheable = "/blobs/sha256:" in url or "/manifests/sha256:" in url or "/manifests/" in url or "hub.docker.com" in url
    key = CACHE / hashlib.sha256((url + "|" + (accept or "")).encode()).hexdigest()
    if cacheable and "auth.docker.io" not in url and key.is_file():
        return key.read_bytes()
    data = _get_uncached(url, token, accept)
    if cacheable and "auth.docker.io" not in url:
        CACHE.mkdir(parents=True, exist_ok=True)
        key.write_bytes(data)
    return data


def _get_uncached(url: str, token: str | None = None, accept: str | None = None) -> bytes:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    if accept:
        request.add_header("Accept", accept)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def token(repo: str) -> str:
    data = _get(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull")
    return json.loads(data)["token"]


def manifest(repo: str, ref: str, tok: str) -> dict:
    return json.loads(_get(f"https://registry-1.docker.io/v2/{repo}/manifests/{ref}", tok, INDEX_TYPES))


def config(repo: str, ref: str, *, arch: str = "amd64") -> tuple[dict, str]:
    tok = token(repo)
    m = manifest(repo, ref, tok)
    digest = None
    if "manifests" in m:
        pick = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == arch
                and x.get("platform", {}).get("os") == "linux"]
        if not pick:
            raise LookupError(f"{repo}:{ref} has no linux/{arch} variant")
        digest = pick[0]["digest"]
        m = manifest(repo, digest, tok)
    blob = json.loads(_get(f"https://registry-1.docker.io/v2/{repo}/blobs/{m['config']['digest']}", tok))
    return blob, digest or ""


def official_config(task: str, tag: str = "20251031") -> dict:
    blob, _ = config("alexgshaw/" + task, tag)
    return blob


def hub_tags(repo: str, prefix: str) -> list[str]:
    url = f"https://hub.docker.com/v2/repositories/{repo}/tags?name={prefix}&page_size=100"
    data = json.loads(_get(url))
    return [r["name"] for r in data.get("results", [])]


def candidate_tags(ref: str, official: dict) -> list[str]:
    name, _, tag = ref.partition(":")
    env = dict(e.split("=", 1) for e in official.get("config", {}).get("Env", []) if "=" in e)
    if name == "python" and "PYTHON_VERSION" in env:
        minor = ".".join(env["PYTHON_VERSION"].split(".")[:2])
        return [tag.replace(minor, env["PYTHON_VERSION"], 1)] if minor in tag else []
    if name == "ubuntu" and tag in ("24.04", "noble"):
        base_created = next((h.get("created", "") for h in official.get("history", []) if "ADD file:" in
                             h.get("created_by", "")), "")
        limit = re.sub(r"[^0-9]", "", base_created[:10]) or "99999999"
        tags = [t for t in hub_tags("library/ubuntu", "noble-20") if re.fullmatch(r"noble-\d{8}(\.\d+)?", t)]
        # Newest dated tag not after the official base layer's creation date first.
        return sorted((t for t in tags if t[6:14] <= limit), reverse=True)
    if name == "debian":
        return [tag] + sorted(hub_tags("library/debian", "trixie-20"))
    return [tag]


def match_base(ref: str, official: dict) -> dict:
    """Find the base tag whose amd64 layers prefix the official image; return its arm64 digest pin."""
    name = ref.split(":")[0]
    repo = name if "/" in name else "library/" + name
    wanted = official["rootfs"]["diff_ids"]
    tried = []
    for tag in candidate_tags(ref, official):
        try:
            amd, _ = config(repo, tag, arch="amd64")
        except Exception as exc:  # noqa: BLE001 - recorded
            tried.append({"tag": tag, "error": str(exc)[:200]})
            continue
        ids = amd["rootfs"]["diff_ids"]
        ok = wanted[:len(ids)] == ids
        tried.append({"tag": tag, "layers": len(ids), "prefix_match": ok})
        if ok:
            arm, arm_digest = config(repo, tag, arch="arm64")
            return {"ref": ref, "match": "layer_prefix", "matched_tag": f"{name}:{tag}",
                    "pinned": f"{name}:{tag}@{arm_digest}",
                    "amd64_layers_matched": len(ids), "arm64_created": arm.get("created"),
                    "arm64_env": [e for e in arm.get("config", {}).get("Env", []) if "VERSION" in e],
                    "candidates_tried": tried}
    env = dict(e.split("=", 1) for e in official.get("config", {}).get("Env", []) if "=" in e)
    if name == "python" and "PYTHON_VERSION" in env and tried and "error" not in tried[0]:
        # Layer digests of re-published python tags do not reproduce; match the CPython release instead.
        tag = tried[0]["tag"]
        arm, arm_digest = config(repo, tag, arch="arm64")
        return {"ref": ref, "match": "python_version_only", "matched_tag": f"{name}:{tag}",
                "pinned": f"{name}:{tag}@{arm_digest}", "official_python_version": env["PYTHON_VERSION"],
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
