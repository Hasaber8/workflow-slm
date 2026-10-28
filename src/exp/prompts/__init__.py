from pathlib import Path
from typing import Optional

from exp.world.aliases import REGION_ALIASES, SERVICE_ALIASES

SOP_FILE = Path(__file__).resolve().parent / "sop.md"
ALIAS_CATALOG_MARKER = "{{ALIAS_CATALOG}}"


def render_sop_prompt(template: str) -> str:
    """Insert public aliases, never scenario labels or expected extractions."""
    if template.count(ALIAS_CATALOG_MARKER) != 1:
        raise ValueError("SOP template must contain exactly one alias catalog marker.")
    sections = []
    for title, aliases in (("Service aliases", SERVICE_ALIASES), ("Region aliases", REGION_ALIASES)):
        rows = [f"### {title}", "", "| Canonical value | Accepted names |", "| :--- | :--- |"]
        for canonical, names in aliases.items():
            rows.append(f"| `{canonical}` | {', '.join(f'`{name}`' for name in names)} |")
        sections.append("\n".join(rows))
    return template.replace(ALIAS_CATALOG_MARKER, "\n\n".join(sections))


def load_sop_prompt(path: Optional[Path] = None) -> str:
    return render_sop_prompt((path or SOP_FILE).read_text(encoding="utf-8"))
