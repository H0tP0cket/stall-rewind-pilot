"""Build-time only: keep files the editable install generated in the project tree.

Native SWE-smith /testbed trees contain these install artifacts (for example a
setuptools_scm version module). New files only (including in-place compiled
extensions, as in the native editable /testbed); tracked files that changed and
build/ directories are reported, never copied.
"""
import hashlib
import json
import os
import shutil
import sys


def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main(clean, work, destination):
    report = {"new": [], "changed_tracked": [], "skipped": []}
    for root, dirs, files in os.walk(work):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, work)
            original = os.path.join(clean, rel)
            if os.path.lexists(original):
                if os.path.isfile(original) and not os.path.islink(original) and digest(original) != digest(path):
                    report["changed_tracked"].append(rel)
                continue
            if rel.split(os.sep)[0] == "build" or name.endswith(".pyc"):
                report["skipped"].append(rel)
                continue
            target = os.path.join(destination, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(path, target, follow_symlinks=False)
            report["new"].append(rel)
    os.makedirs(destination, exist_ok=True)
    with open(os.path.join(os.path.dirname(destination.rstrip("/")), "generated.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    if report["changed_tracked"]:
        sys.exit("editable install modified tracked files: " + ", ".join(report["changed_tracked"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
