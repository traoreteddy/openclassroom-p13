"""Rend un document Markdown de `docs/` en page HTML prête à imprimer en PDF.

    uv run --directory backend --with markdown python ../scripts/render_note.py [docs/x.md]

Sans argument : la note MCP. Puis, dans un navigateur (ou Playwright) : imprimer
`docs/.build/<nom>.html` en A4 vers `docs/<nom>.pdf`. Page de garde (titre, sous-titre en
gras, tableau de métadonnées), sommaire, figures SVG en ligne, captures PNG incorporées ;
le Markdown reste la source unique.
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "docs" / "note-mcp-vision.md"
OUT_DIR = ROOT / "docs" / ".build"
OUT = OUT_DIR / f"{SOURCE.stem}.html"

CSS = """
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.35; color: #1f1f1f; margin: 0; }
h1 { font-size: 19pt; margin: 0 0 6pt; line-height: 1.2; }
h2 { font-size: 13.5pt; margin: 16pt 0 5pt; border-bottom: 1px solid #c9c2b4; padding-bottom: 2pt; page-break-after: avoid; }
h2.break { page-break-before: always; }
h3 { font-size: 11pt; margin: 10pt 0 4pt; page-break-after: avoid; }
p { margin: 3pt 0 5pt; text-align: justify; } li { margin: 1.5pt 0; } ul, ol { margin: 3pt 0 6pt; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 8pt; font-size: 8.4pt; }
tr { page-break-inside: avoid; } thead { display: table-header-group; }
th, td { border: 1px solid #d6d0c4; padding: 2.5pt 4.5pt; vertical-align: top; text-align: left; }
th { background: #f1ede4; } tr:nth-child(even) td { background: #faf8f4; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.6pt; }
hr { border: 0; border-top: 1px solid #c9c2b4; margin: 10pt 0; }

/* page de garde */
.cover { page-break-after: always; display: flex; flex-direction: column; min-height: 240mm; }
.cover__kicker { color: #8a5a2b; font-weight: 700; letter-spacing: .08em; font-size: 9.5pt; text-transform: uppercase; margin-top: 30mm; }
.cover h1 { font-size: 26pt; margin: 8pt 0 10pt; line-height: 1.15; }
.cover__subtitle { font-size: 13pt; color: #444; margin: 0 0 24pt; }
.cover table { width: auto; min-width: 70%; font-size: 10pt; margin-bottom: 26pt; }
.cover th { width: 36mm; }
.toc h2 { border: 0; font-size: 12pt; margin: 0 0 6pt; }
.toc ol { margin: 0; padding-left: 0; list-style: none; columns: 1; }
.toc li { margin: 2pt 0; padding: 2pt 0; border-bottom: 1px dotted #d6d0c4; }
.toc li span { color: #8a5a2b; font-weight: 700; display: inline-block; width: 2.4em; }
.cover__footer { margin-top: auto; color: #6b6b6b; font-size: 9pt; }

/* figures */
figure { margin: 8pt 0 12pt; page-break-inside: avoid; }
figure svg { width: 88%; height: auto; display: block; margin: 0 auto; }
figure img { max-width: 100%; max-height: 150mm; display: block; margin: 0 auto; border: 1px solid #d6d0c4; border-radius: 4pt; }
figure.narrow img { max-width: 62%; }
figcaption { font-size: 8.8pt; color: #555; margin-top: 4pt; text-align: center; font-style: italic; }

/* chiffres clés */
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8pt; margin: 10pt 0 12pt; page-break-inside: avoid; }
.tile { border: 1px solid #d6d0c4; border-top: 3px solid #8a5a2b; border-radius: 6pt; padding: 8pt 8pt 7pt; background: #faf8f4; text-align: center; }
.tile__value { font-size: 15pt; font-weight: 700; color: #8a5a2b; line-height: 1.1; }
.tile__label { font-size: 8.4pt; color: #444; margin-top: 4pt; line-height: 1.3; }

/* résumé */
.summary { background: #f6f1e7; border-left: 4px solid #8a5a2b; padding: 8pt 12pt; margin: 8pt 0 12pt; }
.summary h2 { border: 0; margin-top: 2pt; }
"""


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")

    # Page de garde : titre (h1), sous-titre (ligne en gras), tableau de métadonnées
    title = re.search(r"^# (.+)$", text, re.M).group(1)
    meta_block = re.search(r"^\| \|.*?\n\n", text, re.M | re.S).group(0)
    head = text[: text.index(meta_block)]
    subtitle = " ".join(re.search(r"\*\*(.+?)\*\*", head, re.S).group(1).split())
    meta_html = markdown.markdown(meta_block, extensions=["tables"]).replace(
        "<thead>\n<tr>\n<th></th>\n<th></th>\n</tr>\n</thead>", ""
    )
    meta_html = re.sub(r"<tr>\n<td>(.*?)</td>", r"<tr>\n<th>\1</th>", meta_html)
    body_md = text[text.index(meta_block) + len(meta_block) :]

    # Sommaire à partir des titres de niveau 2
    toc_items = []
    for heading in re.findall(r"^## (.+)$", body_md, re.M):
        number, _, label = heading.partition(". ")
        if not label:
            number, label = "", heading
        toc_items.append(f"<li><span>{number}</span>{label}</li>")

    body = markdown.markdown(body_md, extensions=["tables", "fenced_code", "md_in_html"])

    # Figures SVG en ligne (pas de dépendance au chemin des images dans le PDF)
    def inline_svg(match: re.Match[str]) -> str:
        alt, src = match.group(1), match.group(2)
        path = SOURCE.parent / src
        if path.suffix.lower() == ".svg":
            svg = re.sub(r"<\?xml[^>]*\?>", "", path.read_text(encoding="utf-8"))
            return f"<figure>{svg}<figcaption>{alt}</figcaption></figure>"
        # Capture PNG : incorporée en base64 pour que le HTML soit autonome
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        css_class = ' class="narrow"' if "narrow" in alt.lower() or path.stem.endswith("-n") else ""
        return (
            f'<figure{css_class}><img alt="{alt}" src="data:image/png;base64,{data}">'
            f"<figcaption>{alt}</figcaption></figure>"
        )

    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', inline_svg, body)
    body = body.replace("<h2>Résumé pour décision</h2>", '<div class="summary"><h2>Résumé pour décision</h2>', 1)
    body = body.replace("<h2>1. Contexte", '</div><h2>1. Contexte', 1)
        
    cover = f"""
<section class="cover">
  <div class="cover__kicker">Note de cadrage · système non implémenté</div>
  <h1>{title}</h1>
  <p class="cover__subtitle">{subtitle}</p>
  {meta_html}
  <nav class="toc"><h2>Sommaire</h2><ol>{"".join(toc_items)}</ol></nav>
  <p class="cover__footer">Fédération française des échecs · projet « coach d'ouvertures »</p>
</section>"""

    html = (
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        f"<title>{title}</title><style>{CSS}</style></head><body>{cover}{body}</body></html>"
    )
    OUT_DIR.mkdir(exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"→ {OUT.relative_to(ROOT)} ({len(html) // 1000} ko)")


if __name__ == "__main__":
    main()
