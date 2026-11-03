"""Host W transport with exact recorded modification times.

Python's TarInfo.mtime and extractall use floating-point seconds. PAX textual
times plus explicit ns restoration preserve the precision observable by tasks.
This transports recorded W; it neither invents runtime restoration nor follows
workspace symlinks into the research host.
"""
from __future__ import annotations

from decimal import Decimal
import io
import os
import stat
from pathlib import Path
import tarfile
from typing import Mapping, Any

# Owner read+search: the host-copy projection for a container directory its owner cannot search.
_HOST_SEARCH = stat.S_IRUSR | stat.S_IXUSR


def pack_directory(path: Path, *, root_owned: bool = True,
                   ownership: Mapping[str, Mapping[str, Any]] | None = None,
                   default_owner: tuple[int, int] | None = None) -> bytes:
    path = Path(path)
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for source in sorted(path.rglob("*")):
            info = archive.gettarinfo(str(source), arcname="./" + source.relative_to(path).as_posix())
            ns = source.lstat().st_mtime_ns
            info.pax_headers["mtime"] = str(Decimal(ns) / 1000000000)
            if root_owned:
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
            if default_owner is not None:
                info.uid, info.gid = default_owner
                info.uname = info.gname = ""
            previous = (ownership or {}).get(source.relative_to(path).as_posix())
            if previous is not None:
                info.uid, info.gid = previous["uid"], previous["gid"]
                info.uname = info.gname = ""
                # tar_filter deliberately sanitizes host permissions. Recover
                # the original container mode when the host still has exactly
                # that transport projection; do not install setuid bits on the
                # research host. An explicit adapter mode change is retained.
                if "mode" in previous and (info.mode & 0o7777) == (previous["mode"] & 0o755):
                    info.mode = previous["mode"]
                # extract_directory gives a non-searchable directory with
                # children owner r+x on the host copy only (see there); undo
                # exactly that projection back to the recorded mode.
                elif ("mode" in previous and info.isdir() and not previous["mode"] & stat.S_IXUSR
                      and (info.mode & 0o7777) == ((previous["mode"] & 0o755) | _HOST_SEARCH)):
                    info.mode = previous["mode"]
            if info.isreg():
                with source.open("rb") as data:
                    archive.addfile(info, data)
            else:
                archive.addfile(info)
    return out.getvalue()


def extract_directory(data: bytes, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        # A container directory its owner cannot search (e.g. telecom-entity-resolution's `chmod -R 444 /app/data`,
        # applied as root) leaves its children unreachable to the research user, so their times could not be set and
        # graders and manifests could not read them. Such a directory with children gets owner r+x on this host copy
        # only (pack_directory undoes exactly that projection from the recorded mode); the container-side W keeps
        # its captured mode. Archives without such a directory extract exactly as before.
        parents = {Path(m.name.lstrip("/")).parent for m in archive.getmembers()}

        def safe_member(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo:
            filtered = tarfile.tar_filter(member, destination)
            if filtered.islnk() and not (Path(destination) / filtered.linkname).resolve().is_relative_to(path.resolve()):
                raise ValueError("Workspace hardlink escapes exported workspace")
            if (filtered.isdir() and filtered.mode is not None and not filtered.mode & stat.S_IXUSR
                    and Path(member.name.lstrip("/")) in parents):
                filtered = filtered.replace(mode=filtered.mode | _HOST_SEARCH, deep=False)
            return filtered
        archive.extractall(path, filter=safe_member)
        # Host copies are read by graders and manifests as the research user. A container file its owner cannot
        # read (e.g. ProgramBench's root-owned 0111 reference executable) gets owner-read on this host copy only;
        # contents are unchanged and the container-side W keeps its captured modes.
        for member in archive.getmembers():
            if member.isfile() and not member.mode & stat.S_IRUSR:
                dest = path / member.name.lstrip("/")
                dest.chmod(dest.stat().st_mode | stat.S_IRUSR)
        # Apply directory times last, after every child has been materialized.
        for member in sorted(archive.getmembers(), key=lambda m: len(Path(m.name).parts), reverse=True):
            dest = path / member.name.lstrip("/")
            # A symlink itself may point into R. Never follow it while applying
            # metadata; reject escape through any parent-directory symlink.
            if dest != path and not dest.parent.resolve().is_relative_to(path.resolve()):
                raise ValueError("Workspace metadata path escapes export")
            ns = int(Decimal(member.pax_headers.get("mtime", str(member.mtime))) * 1000000000)
            os.utime(dest, ns=(ns, ns), follow_symlinks=False)
