"""Researcher-only image inventory: path, size and sha256 of every regular file outside W."""
import hashlib
import json
import os
import sys

SKIP = ("/proc", "/sys", "/dev", "/workspace", "/scan")


def main():
    rows = []
    for root, dirs, files in os.walk("/"):
        if any(root == s or root.startswith(s + "/") for s in SKIP):
            dirs[:] = []
            continue
        for name in files:
            path = os.path.join(root, name)
            try:
                if os.path.islink(path) or not os.path.isfile(path):
                    continue
                size = os.path.getsize(path)
                digest = None
                if size <= 50 * 1024 * 1024:
                    with open(path, "rb") as handle:
                        digest = hashlib.sha256(handle.read()).hexdigest()
                rows.append([path, size, digest])
            except OSError as exc:
                rows.append([path, None, "error:" + type(exc).__name__])
    json.dump(rows, sys.stdout)


if __name__ == "__main__":
    main()
