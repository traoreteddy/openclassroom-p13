"""Assemble les schémas SVG de `docs/img/` en une page HTML : un schéma par page A4 paysage.

    uv run --directory backend python ../scripts/render_schemas.py

Puis imprimer `docs/.build/architecture-schemas.html` en PDF (Playwright, `preferCSSPageSize`)
vers `docs/architecture-schemas.pdf`. Ce PDF ne contient que les schémas, sans texte ;
la version complète avec explications est `docs/architecture.pdf` (rendue par `render_note.py`).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
OUT = ROOT / "docs" / ".build" / "architecture-schemas.html"

SCHEMAS = [
    ("Schéma d'ensemble", "Navigateur, pile Docker Compose, modèle local, services tiers", "architecture-app.svg"),
    ("Graphe de l'agent", "Parcours « position » (ordre fixé) et parcours « question » (le modèle choisit)", "graphe-agent.svg"),
]

CSS = """
@page { size: A4 landscape; margin: 12mm 14mm 12mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: #1f1f1f; margin: 0; }
section { page-break-after: always; display: flex; flex-direction: column; height: 182mm; }
section:last-child { page-break-after: auto; }
header { display: flex; align-items: baseline; justify-content: space-between; border-bottom: 1px solid #c9c2b4; padding-bottom: 4pt; margin-bottom: 6pt; }
h1 { font-size: 15pt; margin: 0; }
h1 span { color: #8a5a2b; font-weight: 700; margin-right: 8pt; }
header p { margin: 0; font-size: 9.5pt; color: #555; }
.fig { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; }
.fig svg { width: 100%; height: 100%; max-height: 160mm; }
footer { font-size: 8pt; color: #777; display: flex; justify-content: space-between; margin-top: 4pt; }
"""


def main() -> None:
    sections = []
    for index, (title, subtitle, filename) in enumerate(SCHEMAS, start=1):
        svg = re.sub(r"<\?xml[^>]*\?>", "", (IMG / filename).read_text(encoding="utf-8"))
        sections.append(
            f"<section><header><h1><span>{index}</span>{title}</h1><p>{subtitle}</p></header>"
            f'<div class="fig">{svg}</div>'
            f"<footer><span>Coach d'ouvertures · architecture de l'application · FFE</span>"
            f"<span>{index} / {len(SCHEMAS)}</span></footer></section>"
        )
    html = (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f"<title>Coach d'ouvertures : schémas d'architecture</title><style>{CSS}</style></head>"
        f"<body>{''.join(sections)}</body></html>"
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"→ {OUT.relative_to(ROOT)} ({len(html) // 1000} ko)")


if __name__ == "__main__":
    main()
