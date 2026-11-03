"""Disclosed persistence repair for copied, pinned File Backup references.

This produces an adapted witness, never an unchanged upstream solve. It changes
CP3 full/pack persistence and CP4 full persistence only. CP4's existing whole-pack
classification remains unchanged. Originals and existing destinations are never
overwritten. No expected test output is embedded here.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


ADAPTATION_VERSION = "scb.file_backup.reference-persistence.v1"
SOURCE_REVISION = "ef6a9dd13911566b6b01075ca121758c9f7b5c5f"
EXECUTOR_HASHES = {
    3: "d9b5bc21864c377299693b60776538cf87c665e7724c7d88dca008c4df9b44f3",
    4: "029100ee673a5c69de9b319451a5626aec3c6af74ef06baf17b3e57b059c6235",
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _digest(p.read_bytes())
            for p in sorted(root.rglob("*")) if p.is_file()}


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Pinned source patch anchor occurs {count} times: {old[:80]!r}")
    return text.replace(old, new, 1)


def _repair_executor(text: str, stage: int) -> str:
    text = _replace_once(text, "import hashlib\n", "import hashlib\nimport shutil\n")
    # Pass the existing --backup root through the non-pack factory. No change to
    # verify, omitted-strategy, or no-root behavior.
    text = _replace_once(
        text,
        "_create_strategy_executor(job, emit, run_time)\n",
        "_create_strategy_executor(job, emit, run_time, backup_root)\n",
    )
    text = _replace_once(
        text,
        "def _create_strategy_executor(\n    job: Job,\n    emit: Callable[[dict[str, object]], None],\n    run_time: datetime | None,\n) -> BaseStrategyExecutor | None:\n",
        "def _create_strategy_executor(\n    job: Job,\n    emit: Callable[[dict[str, object]], None],\n    run_time: datetime | None,\n    backup_root: Path | None = None,\n) -> BaseStrategyExecutor | None:\n",
    )
    text = _replace_once(
        text,
        "    strategy = job.strategy\n    if strategy is None:\n        return None\n    if strategy.kind == \"full\":\n        return FullStrategyExecutor(job, emit, run_time)\n",
        "    strategy = job.strategy\n    if strategy is None:\n        return None\n    destination_dir = (\n        backup_root / job.destination_path / job.job_id\n        if backup_root is not None else None\n    )\n    if strategy.kind == \"full\":\n        return FullStrategyExecutor(job, emit, run_time, destination_dir=destination_dir)\n",
    )
    text = _replace_once(
        text,
        "class FullStrategyExecutor(BaseStrategyExecutor):\n    def __init__(\n        self,\n        job: Job,\n        emit: Callable[[dict[str, object]], None],\n        run_time: datetime | None,\n    ) -> None:\n        super().__init__(job, emit, run_time)\n        self._total_size = 0\n",
        "class FullStrategyExecutor(BaseStrategyExecutor):\n    def __init__(\n        self,\n        job: Job,\n        emit: Callable[[dict[str, object]], None],\n        run_time: datetime | None,\n        *,\n        destination_dir: Path | None = None,\n    ) -> None:\n        super().__init__(job, emit, run_time)\n        self._destination_dir = destination_dir\n        self._total_size = 0\n",
    )
    text = _replace_once(
        text,
        '        return "full"\n\n    def handle_file(self, metadata: FileMetadata) -> None:\n        self._total_size += metadata.size\n',
        '        return "full"\n\n    def handle_file(self, metadata: FileMetadata) -> None:\n        if self._destination_dir is not None:\n            target = self._destination_dir / metadata.relative_path\n            target.parent.mkdir(parents=True, exist_ok=True)\n            shutil.copyfile(metadata.absolute_path, target)\n        self._total_size += metadata.size\n',
    )
    if stage == 3:
        text = _replace_once(
            text,
            "        max_bytes: int,\n    ) -> None:\n        super().__init__(job, emit, run_time)\n        self.max_bytes = max_bytes\n",
            "        max_bytes: int,\n        destination_dir: Path | None = None,\n    ) -> None:\n        super().__init__(job, emit, run_time)\n        self._destination_dir = destination_dir\n        self.max_bytes = max_bytes\n",
        )
        text = _replace_once(
            text,
            "        return PackStrategyExecutor(job, emit, run_time, max_bytes=max_bytes)\n",
            "        return PackStrategyExecutor(\n            job, emit, run_time, max_bytes=max_bytes, destination_dir=destination_dir\n        )\n",
        )
        text = _replace_once(
            text,
            '        pack_name = f"pack-{self._pack_id}.tar"\n        payload = {\n',
            '        pack_name = f"pack-{self._pack_id}.tar"\n        if self._destination_dir is not None:\n            self._destination_dir.mkdir(parents=True, exist_ok=True)\n            shutil.copyfile(pack_path, self._destination_dir / pack_name)\n        payload = {\n',
        )
    compile(text, "scheduler/executor.py", "exec")
    return text


def adapt_reference(source: Path, destination: Path, stage: int) -> dict:
    """Copy a checkpoint reference and repair explicit destination persistence.

    ``source`` must be the upstream ``solutions/checkpoint_3`` or ``checkpoint_4``
    directory. The executor hash must match the audited immutable revision.
    ``destination`` must not exist and must not overlap the source tree.
    The returned provenance binds every original/copied file and changed hash.
    """
    if isinstance(stage, bool) or stage not in EXECUTOR_HASHES:
        raise ValueError("Only checkpoint 3 or 4 references have this persistence repair")
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite reference destination: {destination}")
    destination = destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination trees must not overlap")
    if not source.is_dir() or any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Reference source must be a directory without symlinks")
    original = _files(source)
    relative = "scheduler/executor.py"
    if original.get(relative) != EXECUTOR_HASHES[stage]:
        raise ValueError("Reference executor does not match the pinned stage/source revision")
    rewritten = _repair_executor((source / relative).read_text(), stage)
    # Validate the complete patch before creating any output. copytree refuses an
    # already-existing destination even if another process creates it meanwhile.
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    (destination / relative).write_text(rewritten)
    adapted = _files(destination)
    if _files(source) != original:
        raise RuntimeError("Reference source changed during adaptation")
    changed = [p for p in original if original[p] != adapted[p]]
    if changed != [relative] or set(original) != set(adapted):
        raise RuntimeError("Unexpected reference adaptation file changes")
    description = [
        "Disclosed adapted witness; not an unchanged upstream reference solve.",
        "Full strategy writes selected new/changed source bytes beneath supplied --backup destination.",
        "Verify/no-strategy and omitted --backup retain existing non-writing behavior.",
    ]
    if stage == 3:
        description.append("Pack strategy persists exactly the generated tar bytes beneath supplied --backup destination; it still performs no incremental pack comparison.")
    else:
        description.append("Existing CP4 persistent pack writing and whole-pack classification are unchanged; no stale-cleanup obligation is added to the public task.")
    return {
        "adaptation_version": ADAPTATION_VERSION,
        "source_revision": SOURCE_REVISION,
        "stage": stage,
        "source": str(source),
        "destination": str(destination),
        "source_preserved": True,
        "changed_files": [{"path": p, "before_sha256": original[p], "after_sha256": adapted[p]} for p in changed],
        "original_files": original,
        "adapted_files": adapted,
        "description": description,
    }
