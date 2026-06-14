# -*- coding: utf-8 -*-
"""RELAZIONE_v2.md -> relazione_v2.html (stile di stampa) -> RELAZIONE_v2.pdf via Edge headless."""
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
MD = ROOT / "RELAZIONE_v2.md"
HTML = ROOT / "relazione_v2.html"
PDF = ROOT / "RELAZIONE_v2.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #1c2128;
  max-width: 175mm; margin: 0 auto;
}
h1 { font-size: 19pt; color: #16324a; border-bottom: 2.5px solid #2d6a8f;
     padding-bottom: 4px; margin: 0 0 10px; page-break-before: always; }
body > h1:first-of-type { page-break-before: avoid; text-align: center;
     border-bottom: none; font-size: 24pt; margin-top: 30mm; }
body > h1:first-of-type + h2 { text-align: center; border: none; color: #44505c;
     font-size: 13pt; font-weight: 500; line-height: 1.5; }
h2 { font-size: 13.5pt; color: #1f4e6e; margin: 22px 0 8px; page-break-after: avoid; }
h3 { font-size: 11.5pt; color: #2d6a8f; margin: 16px 0 6px; page-break-after: avoid; }
p { margin: 6px 0; text-align: justify; }
strong { color: #16324a; }
a { color: #2d6a8f; text-decoration: none; }
blockquote { border-left: 4px solid #2d6a8f; background: #f2f6f9;
  margin: 10px 0; padding: 8px 14px; color: #243240; page-break-inside: avoid; }
blockquote p { text-align: left; }
code { font-family: "Cascadia Mono", Consolas, monospace; font-size: 9pt;
  background: #eef1f4; padding: 1px 4px; border-radius: 3px; }
pre { background: #f4f6f8; border: 1px solid #dde3e8; border-radius: 6px;
  padding: 10px 12px; overflow-x: hidden; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; margin: 10px 0;
  font-size: 9pt; page-break-inside: avoid; }
th { background: #2d6a8f; color: #fff; padding: 5px 7px; text-align: left; }
td { border-bottom: 1px solid #dde3e8; padding: 4px 7px; }
tr:nth-child(even) td { background: #f6f8fa; }
img { max-width: 100%; display: block; margin: 12px auto 4px;
  page-break-inside: avoid; border: 1px solid #e3e8ec; border-radius: 4px; }
img + em, p > em:only-child { display: block; text-align: center; font-size: 9pt;
  color: #55606b; margin: 2px auto 12px; max-width: 90%; }
/* i --- del markdown precedono sempre un h1 che gia' apre pagina nuova:
   visibili resterebbero orfani su pagine quasi vuote */
hr { display: none; }
ul, ol { margin: 6px 0; padding-left: 22px; }
li { margin: 3px 0; }
"""

text = MD.read_text(encoding="utf-8")
body = markdown.markdown(
    text, extensions=["tables", "fenced_code", "toc", "sane_lists"],
    extension_configs={"toc": {"permalink": False}},
)
HTML.write_text(
    "<!DOCTYPE html><html lang='it'><head><meta charset='utf-8'>"
    f"<title>Da OCGAN a PatchCore — Relazione</title><style>{CSS}</style></head>"
    f"<body>{body}</body></html>",
    encoding="utf-8",
)
print(f"html ok: {HTML} ({HTML.stat().st_size} bytes)")

cmd = [
    EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
    f"--print-to-pdf={PDF}", HTML.as_uri(),
]
res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
if PDF.exists():
    print(f"pdf ok: {PDF} ({PDF.stat().st_size} bytes)")
else:
    print("PDF NON GENERATO", res.returncode, res.stderr[-500:] if res.stderr else "")
    sys.exit(1)
