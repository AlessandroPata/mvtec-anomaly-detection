#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_pdf.py — genera RELAZIONE.pdf da RELAZIONE_v2.md
Uso:  python build_pdf.py
Requisiti:  pip install weasyprint markdown
            (font consigliati: Caladea + Carlito, pacchetti fonts-crosextra-*)
Le figure vengono cercate in ./figures/ : se un file manca, nel PDF compare
un segnaposto. Rigenera le figure con gen_figures.py e rilancia lo script.
"""
import os, re, html
import markdown
from weasyprint import HTML

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "RELAZIONE_v2.md")
OUT = os.path.join(HERE, "RELAZIONE.pdf")

INK     = "#15212e"   # blu inchiostro
ACCENT  = "#c2492e"   # rosso "anomalia"
MUTED   = "#5d6b77"
PANEL   = "#f3f5f7"
HAIR    = "#d4dade"

# ---------------------------------------------------------------- preprocess
with open(SRC, encoding="utf-8") as f:
    md_text = f.read()

# separa metadati di testata (titolo, sottotitolo, autore...) dal corpo
parts = md_text.split("\n---\n")
head_block = parts[0]
rest = "\n---\n".join(parts[1:])

m_title  = re.search(r"^# (.+)$", head_block, re.M)
m_sub    = re.search(r"^## (.+)$", head_block, re.M)
title    = m_title.group(1).strip() if m_title else "Relazione"
subtitle = m_sub.group(1).strip() if m_sub else ""
meta = dict(re.findall(r"\*\*(.+?):\*\*\s*(.+)", head_block))

# rimuove l'indice markdown (il PDF ha il suo sommario con numeri di pagina)
rest = re.sub(r"## Indice.*?\n---\n", "", rest, count=1, flags=re.S)

# figura + didascalia -> blocco <figure> (segnaposto se il file manca)
def figure_repl(m):
    alt, path, caption = m.group(1), m.group(2), m.group(3)
    cap_html = re.sub(r"\\\*", "*", caption)
    if os.path.exists(os.path.join(HERE, path)):
        body = f'<img src="{path}" alt="{html.escape(alt)}"/>'
    else:
        body = (f'<div class="fig-missing">Figura non ancora generata — '
                f'inserire <code>{html.escape(path)}</code> '
                f'(da <code>gen_figures.py</code>) e rilanciare il build</div>')
    return (f'<figure>{body}<figcaption>{cap_html}</figcaption></figure>\n')

rest = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)\n\*(.+?)\*\n",
              figure_repl, rest)

# id stabili per i capitoli (per il sommario)
def heading_id(txt):
    s = re.sub(r"[^a-z0-9]+", "-", txt.lower()).strip("-")
    return "sec-" + s[:60]

chapters = []   # (id, titolo)
def h1_repl(m):
    t = m.group(1).strip()
    hid = heading_id(t)
    chapters.append((hid, t))
    return f'\n<h1 id="{hid}">{t}</h1>\n'
rest = re.sub(r"^# (.+)$", h1_repl, rest, flags=re.M)

# markdown -> html
body_html = markdown.markdown(
    rest, extensions=["tables", "fenced_code", "attr_list"])

# le didascalie dentro <figure> non vanno reincapsulate
body_html = body_html.replace("<p><figure>", "<figure>").replace("</figure></p>", "</figure>")

# sommario
toc_items = "\n".join(
    f'<li><a href="#{hid}">{html.escape(t)}</a></li>' for hid, t in chapters)

# motivo grafico "patch grid": griglia di quadratini, uno acceso (l'anomalia)
def patch_grid(n, cell, gap, anomaly, base, accent, op="1"):
    cells = []
    for r in range(n):
        for c in range(n):
            col = accent if (r, c) == anomaly else base
            cells.append(f'<rect x="{c*(cell+gap)}" y="{r*(cell+gap)}" '
                         f'width="{cell}" height="{cell}" rx="1.5" fill="{col}" fill-opacity="{op}"/>')
    side = n*(cell+gap)-gap
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side} {side}" '
            f'width="{side}" height="{side}">' + "".join(cells) + "</svg>")

cover_grid = patch_grid(7, 16, 6, (2, 4), "#dde3e8", ACCENT)

stats = """
<div class="cover-stats">
  <div><span class="n">11.738</span><span class="l">run di ablation su disco</span></div>
  <div><span class="n">15</span><span class="l">categorie MVTec&nbsp;AD</span></div>
  <div><span class="n">0.9846</span><span class="l">macro AUROC finale</span></div>
</div>"""

cover = f"""
<section class="cover">
  <div class="cover-kicker">Relazione di progetto · Anomaly detection one-class</div>
  <div class="cover-motif">{cover_grid}</div>
  <h1 class="cover-title">{html.escape(title)}</h1>
  <p class="cover-sub">{html.escape(subtitle)}</p>
  {stats}
  <div class="cover-meta">
    <p><strong>{html.escape(meta.get('Autore','')).strip()}</strong></p>
    <p>{html.escape(meta.get('Data','')).strip()}</p>
    <p class="code-line">{meta.get('Codice','')}</p>
  </div>
</section>
<section class="toc">
  <h2>Sommario</h2>
  <ul>{toc_items}</ul>
</section>
"""

css = f"""
@page {{
  size: A4;
  margin: 24mm 20mm 22mm 20mm;
  @top-left {{ content: string(doctitle); font-family: Carlito; font-size: 8pt;
               letter-spacing: .6pt; text-transform: uppercase; color: {MUTED}; }}
  @top-right {{ content: string(chapter); font-family: Carlito; font-size: 8pt;
                color: {MUTED}; font-style: italic; }}
  @bottom-center {{ content: counter(page); font-family: Carlito;
                    font-size: 9pt; color: {MUTED}; }}
}}
@page :first {{ margin: 0; @top-left {{content:none}} @top-right {{content:none}}
                @bottom-center {{content:none}} }}
@page toc {{ @top-right {{ content: "Sommario"; }} }}

html {{ string-set: doctitle "Da OCGAN a PatchCore"; }}
body {{ font-family: Caladea, Georgia, serif; font-size: 10.3pt; line-height: 1.5;
        color: #20262c; text-align: justify; hyphens: auto; }}

/* ------------------------------------------------ copertina */
.cover {{ page-break-after: always; height: 257mm; padding: 30mm 24mm 24mm 24mm;
          text-align: left; position: relative; }}
.cover-kicker {{ font-family: Carlito; font-size: 10pt; letter-spacing: 2pt;
                 text-transform: uppercase; color: {ACCENT}; font-weight: bold;
                 max-width: 118mm; }}
.cover-motif {{ position: absolute; top: 26mm; right: 24mm; }}
.cover-title {{ font-size: 33pt; line-height: 1.12; color: {INK};
                margin: 26mm 0 8mm 0; font-weight: bold; max-width: 130mm;
                page-break-before: avoid; border-bottom: none; padding-bottom: 0;
                string-set: none; }}
.cover-meta {{ position: absolute; bottom: 26mm; left: 24mm; }}
.cover-sub {{ font-size: 13pt; line-height: 1.45; color: #3a4550; font-style: italic;
              max-width: 140mm; margin-bottom: 18mm; text-align: left; }}
.cover-stats {{ display: flex; gap: 14mm; border-top: .6pt solid {HAIR};
                border-bottom: .6pt solid {HAIR}; padding: 7mm 0; margin-bottom: 16mm; }}
.cover-stats div {{ display: block; }}
.cover-stats .n {{ display: block; font-family: Carlito; font-size: 22pt;
                   font-weight: bold; color: {INK}; }}
.cover-stats .l {{ display: block; font-family: Carlito; font-size: 9pt;
                   color: {MUTED}; margin-top: 1mm; }}
.cover-meta p {{ margin: 0 0 1.5mm 0; font-size: 11pt; text-align: left; }}
.cover-meta .code-line {{ font-family: "DejaVu Sans Mono", monospace;
                          font-size: 8.5pt; color: {MUTED}; }}

/* ------------------------------------------------ sommario */
.toc {{ page: toc; page-break-after: always; }}
.toc h2 {{ font-size: 20pt; color: {INK}; margin: 0 0 8mm 0; }}
.toc ul {{ list-style: none; padding: 0; margin: 0; }}
.toc li {{ margin: 0 0 3.2mm 0; font-family: Carlito; font-size: 11pt; }}
.toc a {{ text-decoration: none; color: #20262c; }}
.toc a::after {{ content: leader(". ") target-counter(attr(href), page);
                 color: {MUTED}; font-size: 10pt; }}

/* ------------------------------------------------ titoli */
h1 {{ string-set: chapter content(); page-break-before: always;
      font-size: 19pt; line-height: 1.2; color: {INK};
      margin: 0 0 6mm 0; padding-bottom: 3mm; border-bottom: .8pt solid {HAIR};
      text-align: left; }}
h2 {{ font-size: 13pt; color: {INK}; margin: 7mm 0 2.5mm 0; text-align: left;
      page-break-after: avoid; }}
h3 {{ font-size: 11pt; color: {INK}; font-style: italic; margin: 5mm 0 2mm 0;
      text-align: left; page-break-after: avoid; }}

p {{ margin: 0 0 2.6mm 0; orphans: 3; widows: 3; }}
strong {{ color: {INK}; }}
a {{ color: {INK}; }}

/* ------------------------------------------------ tabelle */
table {{ width: 100%; border-collapse: collapse; margin: 3.5mm 0 4.5mm 0;
         font-family: Carlito; font-size: 8.3pt; line-height: 1.3;
         page-break-inside: auto; }}
th {{ background: {INK}; color: #fff; font-weight: bold; padding: 1.8mm 2mm;
      text-align: left; }}
td {{ padding: 1.5mm 2mm; border-bottom: .4pt solid {HAIR}; text-align: left;
      vertical-align: top; }}
tr:nth-child(even) td {{ background: #f7f9fa; }}
thead {{ display: table-header-group; }}

/* ------------------------------------------------ codice */
code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.3pt;
        background: {PANEL}; padding: 0 .35mm; border-radius: 1pt; }}
pre {{ background: {PANEL}; border: .4pt solid {HAIR}; border-radius: 2pt;
       padding: 3mm 3.5mm; margin: 3.5mm 0 4.5mm 0; page-break-inside: avoid;
       font-size: 7.8pt; line-height: 1.35; white-space: pre; overflow: hidden;
       text-align: left; }}
pre code {{ background: none; padding: 0; font-size: inherit; }}

/* ------------------------------------------------ citazioni / box */
blockquote {{ background: #fdf4f1; border: .5pt solid #ecd2c9; border-radius: 2pt;
              margin: 4mm 0; padding: 3mm 4.5mm; page-break-inside: avoid;
              text-align: left; }}
blockquote p {{ margin: 0 0 1.5mm 0; }}
blockquote p:last-child {{ margin: 0; }}

/* ------------------------------------------------ figure */
figure {{ margin: 4.5mm 0 5mm 0; text-align: center; page-break-inside: avoid; }}
figure img {{ max-width: 100%; max-height: 105mm; }}
figcaption {{ font-family: Carlito; font-size: 8.6pt; color: {MUTED};
              margin-top: 2mm; text-align: center; line-height: 1.35;
              padding: 0 8mm; }}
.fig-missing {{ border: 1pt dashed #c4ccd3; border-radius: 2pt; color: {MUTED};
                font-family: Carlito; font-size: 9pt; padding: 12mm 8mm;
                background: #fafbfc; }}

ul, ol {{ margin: 1mm 0 3mm 0; padding-left: 6mm; text-align: left; }}
li {{ margin-bottom: 1.2mm; }}
hr {{ border: none; margin: 0; }}
"""

html_doc = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8"><style>{css}</style></head>
<body>{cover}{body_html}</body></html>"""

HTML(string=html_doc, base_url=HERE).write_pdf(OUT)
print("OK ->", OUT)
