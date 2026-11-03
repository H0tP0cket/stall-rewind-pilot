"""Researcher-only: report image files containing any withheld-test line (exact stripped-line match)."""
import json
import os
import sys

SKIP = ("/proc", "/sys", "/dev", "/workspace", "/scan")


def main(path):
    needles = json.load(open(path))
    wanted = {}
    for owner, lines in needles.items():
        for line in lines:
            wanted.setdefault(line, set()).add(owner)
    hits, scanned = [], 0
    for root, dirs, files in os.walk("/"):
        if any(root == s or root.startswith(s + "/") for s in SKIP):
            dirs[:] = []
            continue
        for name in files:
            full = os.path.join(root, name)
            try:
                if os.path.islink(full) or not os.path.isfile(full) or os.path.getsize(full) > 5 * 1024 * 1024:
                    continue
                with open(full, "rb") as handle:
                    text = handle.read().decode("utf-8", "ignore")
            except OSError:
                continue
            scanned += 1
            found = {line.strip() for line in text.splitlines()} & wanted.keys()
            if found:
                for owner in sorted(set().union(*(wanted[f] for f in found))):
                    mine = sorted(f for f in found if owner in wanted[f])
                    hits.append({"path": full, "owner": owner, "count": len(mine), "examples": mine[:3]})
    json.dump({"files_scanned": scanned, "needles": len(wanted), "hits": hits}, sys.stdout)


if __name__ == "__main__":
    main(sys.argv[1])
