"""Conservative local Python source closure for admission fixture receipts.

Dynamic evaluator scripts must still be declared by the task manifest. Keep a
snapshot of the complete package alongside this smaller checked closure, so a
later audit can examine source bytes without rewriting historical bindings.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
from typing import Iterable


def snapshot_source_tree(root: Path, destination: Path) -> dict[str, str]:
    """Retain all local Python bytes before execution, including dynamic helpers."""
    root, destination = Path(root).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    inventory = {}
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Implementation source escaped its package")
        relative = path.relative_to(root)
        data = path.read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        inventory[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    return inventory


def imported_local_sources(root: Path) -> tuple[str, ...]:
    """In-process Python modules; private subprocess scripts need manifest binding."""
    root = Path(root).resolve()
    found = set()
    for value in tuple(sys.modules.values()):
        filename = getattr(value, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(root):
            found.add(path.relative_to(root).as_posix())
    return tuple(sorted(found))


def source_closure(root: Path, initial: Iterable[str]) -> tuple[str, ...]:
    root = Path(root).resolve()
    pending, found = set(initial), set()

    def module(parts: list[str]) -> None:
        for count in range(len(parts) + 1):
            init = Path(*parts[:count]) / "__init__.py"
            if (root / init).is_file():
                pending.add(init.as_posix())
        if parts:
            path = Path(*parts).with_suffix(".py")
            if (root / path).is_file():
                pending.add(path.as_posix())

    while pending:
        name = pending.pop()
        if name in found:
            continue
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".py":
            raise ValueError("Source closure requires package-relative Python files")
        path = (root / rel).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Source dependency escapes implementation root")
        found.add(rel.as_posix())
        module(list(rel.parent.parts) if rel.parent != Path(".") else [])
        package_parts = list(rel.parent.parts) if rel.parent != Path(".") else []
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if parts[0] == "trajectory_retention":
                        module(parts[1:])
            elif isinstance(node, ast.ImportFrom):
                parts = node.module.split(".") if node.module else []
                if node.level:
                    if node.level - 1 > len(package_parts):
                        raise ValueError("Relative import leaves implementation package")
                    parts = package_parts[:len(package_parts) - node.level + 1] + parts
                elif parts and parts[0] == "trajectory_retention":
                    parts = parts[1:]
                else:
                    continue
                module(parts)
                for alias in node.names:
                    if alias.name != "*":
                        module(parts + alias.name.split("."))
    return tuple(sorted(found))
