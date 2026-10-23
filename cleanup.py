#!/usr/bin/env python3
import os
from pathlib import Path

# Remove all monitoring files
root = Path("e:/PROJECT/Hu_hong_duong")

files_to_remove = [
    root / "scripts/train_monitor.py",
    root / "scripts/dashboard_server.py",
    root / "scripts/launch_monitoring.py",
    root / "scripts/QUICKSTART.py",
    root / "scripts/CHEATSHEET.py",
    root / "TRAINING_MONITOR_README.md",
    root / "MONITORING_SETUP_SUMMARY.md",
    root / "MONITORING_INDEX.md",
    root / "START_HERE.py",
    root / "SYSTEM_READY.md",
]

for f in files_to_remove:
    try:
        if f.exists():
            f.unlink()
            print(f"✓ Deleted: {f.name}")
    except Exception as e:
        print(f"✗ Failed: {f.name} - {e}")

print("\n✅ Cleanup complete!")
