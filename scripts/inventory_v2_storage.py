"""Measure the Codexa v3 preparation volume without changing stored artifacts."""

import argparse
import json
import os
import shutil
from pathlib import Path

from codexa.data.production import atomic
from codexa.data.v2_storage import Storage


def inventory(root: Path) -> dict[str, object]:
    storage = Storage.load(root)
    storage.check()
    seen: set[tuple[int, int]] = set()
    components: dict[str, dict[str, int]] = {}
    symlinks = 0
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink():
                symlinks += 1
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue  # A concurrent atomic write was renamed.
            component = path.relative_to(root).parts[0]
            item = components.setdefault(
                component, {"allocated_bytes": 0, "logical_bytes": 0, "files": 0}
            )
            item["files"] += 1
            item["logical_bytes"] += stat.st_size
            key = stat.st_dev, stat.st_ino
            if key not in seen:
                seen.add(key)
                item["allocated_bytes"] += stat.st_blocks * 512
    usage = shutil.disk_usage(root)
    total = sum(x["allocated_bytes"] for x in components.values())
    if total != storage.usage():
        # Active acquisitions can change the tree between two scans; never call this an exact audit.
        consistency = "changing_concurrently"
    else:
        consistency = "same_allocated_total_on_two_scans"
    return {
        "root": str(root),
        "volume_uuid": storage.uuid,
        "free_bytes": usage.free,
        "minimum_free_bytes": storage.reserve,
        "root_allocated_bytes": total,
        "root_maximum_bytes": storage.maximum,
        "raw_maximum_bytes": storage.raw_maximum,
        "components": components,
        "symlink_files_skipped": symlinks,
        "hardlinked_inodes_counted_once": True,
        "consistency": consistency,
        "scope": "Allocated blocks on the mounted volume; concurrent writes may change a second scan. No source file is deleted or modified.",
        "training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root)
    atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
