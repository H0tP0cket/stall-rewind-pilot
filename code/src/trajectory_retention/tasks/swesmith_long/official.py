"""Read the official SWE-smith x86_64 environment's pinned packages without running it.

Layers are streamed from Docker Hub and only package metadata is extracted.
Nothing from the official image enters a worker image except these pins.
"""
from __future__ import annotations

import email.parser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import urllib.request

REGISTRY = "https://registry-1.docker.io/v2/"
ACCEPT = ", ".join(["application/vnd.docker.distribution.manifest.v2+json",
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.oci.image.index.v1+json"])
ENV_PREFIX = "opt/miniconda3/envs/testbed/"


def _token(repository: str) -> str:
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["token"]


def _get(url: str, token: str, accept: str = ACCEPT):
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token, "Accept": accept})
    return urllib.request.urlopen(request, timeout=600)


def manifest(image: str) -> dict:
    repository, _, tag = image.partition(":")
    token = _token(repository)
    with _get(f"{REGISTRY}{repository}/manifests/{tag or 'latest'}", token) as response:
        body = response.read()
        digest = response.headers.get("Docker-Content-Digest")
    data = json.loads(body)
    return {"repository": repository, "tag": tag or "latest", "manifest_digest": digest,
            "manifest_sha256": hashlib.sha256(body).hexdigest(), "manifest": data, "token": token}


def _wanted(name: str) -> bool:
    name = name.lstrip("./")
    parts = PurePosixPath(name).parts
    if (parts and parts[0] == "testbed" and len(parts) in (3, 4) and parts[-1] == "PKG-INFO"
            and parts[-2].endswith(".egg-info")):
        return True  # legacy develop-mode project metadata lives in the source tree
    if not name.startswith(ENV_PREFIX):
        return False
    rest = name[len(ENV_PREFIX):]
    if rest.startswith("conda-meta/") and rest.endswith(".json"):
        return True
    parts = PurePosixPath(rest).parts
    return (len(parts) >= 5 and parts[0] == "lib" and parts[2] == "site-packages"
            and (parts[3].endswith(".dist-info") or parts[3].endswith(".egg-info"))
            and parts[4] in ("METADATA", "PKG-INFO", "direct_url.json", "INSTALLER", "top_level.txt", "RECORD")) or (
            len(parts) == 4 and parts[0] == "lib" and parts[2] == "site-packages"
            and (parts[3].endswith(".pth") or parts[3].endswith(".egg-link")))


def extract_metadata(image: str, destination: Path) -> dict:
    """Stream every layer in order; later layers override earlier files."""
    info = manifest(image)
    layers = info["manifest"].get("layers")
    if not layers:
        raise ValueError("Expected a single-platform image manifest")
    destination.mkdir(parents=True, exist_ok=True)
    seen = []
    for layer in layers:
        with _get(f"{REGISTRY}{info['repository']}/blobs/{layer['digest']}", info["token"], accept="*/*") as response:
            with tarfile.open(fileobj=response, mode="r|*") as archive:
                for member in archive:
                    name = member.name.lstrip("./")
                    base = PurePosixPath(name).name
                    if base.startswith(".wh."):
                        target = destination / str(PurePosixPath(name).parent) / base[4:]
                        if target.exists() and target.is_file():
                            target.unlink()
                        continue
                    if not member.isfile() or not _wanted(name):
                        continue
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    handle = archive.extractfile(member)
                    target.write_bytes(handle.read() if handle else b"")
        seen.append({"digest": layer["digest"], "size": layer["size"]})
    return {"image": image, "manifest_digest": info["manifest_digest"], "manifest_sha256": info["manifest_sha256"],
            "layers": seen}


def installed_packages(root: Path) -> dict:
    """Name/version pins plus editable/direct-url evidence from extracted metadata."""
    env = root / ENV_PREFIX
    python = sorted(p.name for p in (env / "conda-meta").glob("python-3*.json"))
    version = None
    if python:
        version = json.loads((env / "conda-meta" / python[0]).read_text())["version"]
    packages = []
    for meta_dir in sorted((env / "lib").glob("python*/site-packages/*")):
        if not (meta_dir.name.endswith(".dist-info") or meta_dir.name.endswith(".egg-info")):
            continue
        meta = meta_dir / ("METADATA" if meta_dir.name.endswith(".dist-info") else "PKG-INFO")
        if not meta.is_file():
            continue
        parsed = email.parser.Parser().parsestr(meta.read_text(errors="replace"), headersonly=True)
        direct = meta_dir / "direct_url.json"
        direct_url = json.loads(direct.read_text()) if direct.is_file() else None
        installer = (meta_dir / "INSTALLER").read_text().strip() if (meta_dir / "INSTALLER").is_file() else None
        packages.append({"name": parsed["Name"], "version": parsed["Version"], "metadata_dir": meta_dir.name,
                         "installer": installer, "direct_url": direct_url,
                         "editable": bool(direct_url and direct_url.get("dir_info", {}).get("editable"))})
    pth = {p.name: p.read_text(errors="replace") for p in sorted((env / "lib").glob("python*/site-packages/*.pth"))}
    links = sorted((env / "lib").glob("python*/site-packages/*.egg-link"))
    if links and not any(p["editable"] for p in packages):
        # Legacy `setup.py develop` install: egg-link + easy-install.pth -> /testbed.
        if len(links) != 1:
            raise ValueError("Several legacy editable projects")
        link = links[0]
        infos = sorted(root.glob("testbed/*.egg-info/PKG-INFO")) + sorted(root.glob("testbed/*/*.egg-info/PKG-INFO"))
        project_version = None
        name = link.name.removesuffix(".egg-link")
        for info in infos:
            parsed = email.parser.Parser().parsestr(info.read_text(errors="replace"), headersonly=True)
            if normalize(parsed["Name"]) == normalize(name):
                name, project_version = parsed["Name"], parsed["Version"]
        packages.append({"name": name, "version": project_version, "metadata_dir": link.name, "installer": "legacy-develop",
                         "direct_url": None, "editable": True, "egg_link": link.read_text(errors="replace")})
    return {"python_version": version, "packages": packages, "pth_files": pth}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()
