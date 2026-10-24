import datetime
from pathlib import Path

RUNLOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "docs" / "RUNLOG.md"


def log_command(cmd: str, duration_sec: float, outcome: str, notes: str = ""):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"- {ts} | `{cmd}` | {duration_sec:.1f}s | {outcome} | {notes}\n"
    with open(RUNLOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
