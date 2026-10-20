"""ReportLab building blocks: fonts, styles, page template with table of contents, tiles,
callouts, tables, code excerpts and simple vector diagrams."""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ palette
ACCENT = colors.HexColor("#4F46E5")
ACCENT_SOFT = colors.HexColor("#EEF0FD")
ACCENT_LINE = colors.HexColor("#C9CCF6")
INK = colors.HexColor("#14171C")
INK2 = colors.HexColor("#3C4350")
MUTED = colors.HexColor("#677180")
RULE = colors.HexColor("#E2E6EB")
ZEBRA = colors.HexColor("#F6F7F9")
CODE_BG = colors.HexColor("#F4F5F7")
OLD = colors.HexColor("#EB6834")
OLD_SOFT = colors.HexColor("#FDEEE7")
OK = colors.HexColor("#1A7F37")
OK_SOFT = colors.HexColor("#E8F5EC")
WARN_SOFT = colors.HexColor("#FFF6E0")
WARN = colors.HexColor("#9A6700")
GRAY_SOFT = colors.HexColor("#F1F3F6")

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

# ------------------------------------------------------------------ fonts
FONT, BOLD, SEMI, ITALIC, MONO = "Helvetica", "Helvetica-Bold", "Helvetica-Bold", "Helvetica-Oblique", "Courier"


def register_fonts() -> None:
    global FONT, BOLD, SEMI, ITALIC, MONO
    win = Path("C:/Windows/Fonts")
    files = {"Segoe": "segoeui.ttf", "Segoe-Bold": "segoeuib.ttf", "Segoe-Italic": "segoeuii.ttf",
             "Segoe-BoldItalic": "segoeuiz.ttf", "Segoe-Semibold": "seguisb.ttf", "Consolas": "consola.ttf",
             "Consolas-Bold": "consolab.ttf"}
    if not all((win / f).exists() for f in files.values()):
        return
    for name, f in files.items():
        pdfmetrics.registerFont(TTFont(name, str(win / f)))
    from reportlab.lib.fonts import addMapping
    addMapping("Segoe", 0, 0, "Segoe")
    addMapping("Segoe", 1, 0, "Segoe-Bold")
    addMapping("Segoe", 0, 1, "Segoe-Italic")
    addMapping("Segoe", 1, 1, "Segoe-BoldItalic")
    addMapping("Consolas", 0, 0, "Consolas")
    addMapping("Consolas", 1, 0, "Consolas-Bold")
    addMapping("Consolas", 0, 1, "Consolas")
    addMapping("Consolas", 1, 1, "Consolas-Bold")
    FONT, BOLD, SEMI, ITALIC, MONO = "Segoe", "Segoe-Bold", "Segoe-Semibold", "Segoe-Italic", "Consolas"


register_fonts()

# ------------------------------------------------------------------ styles
S: dict[str, ParagraphStyle] = {}


def _styles() -> None:
    base = dict(fontName=FONT, textColor=INK, leading=13.6, fontSize=9.4)
    S["body"] = ParagraphStyle("body", spaceAfter=6, **base)
    S["lead"] = ParagraphStyle("lead", parent=S["body"], fontSize=10.6, leading=15.5, textColor=INK2, spaceAfter=10)
    S["small"] = ParagraphStyle("small", parent=S["body"], fontSize=8.2, leading=11.2, textColor=MUTED, spaceAfter=4)
    S["caption"] = ParagraphStyle("caption", parent=S["small"], alignment=TA_LEFT, spaceBefore=3, spaceAfter=10)
    S["h1"] = ParagraphStyle("h1", fontName=BOLD, fontSize=19, leading=24, textColor=INK, spaceBefore=4,
                             spaceAfter=4)
    S["title"] = ParagraphStyle("title", fontName=BOLD, fontSize=25, leading=31, textColor=INK, spaceAfter=4)
    S["title2"] = ParagraphStyle("title2", fontName=BOLD, fontSize=19, leading=24, textColor=INK, spaceAfter=4)
    S["kicker"] = ParagraphStyle("kicker", fontName=SEMI, fontSize=8.5, leading=11, textColor=ACCENT, spaceAfter=2)
    S["h2"] = ParagraphStyle("h2", fontName=BOLD, fontSize=12.6, leading=16, textColor=INK, spaceBefore=12,
                             spaceAfter=5)
    S["h3"] = ParagraphStyle("h3", fontName=SEMI, fontSize=10.2, leading=13.5, textColor=INK, spaceBefore=8,
                             spaceAfter=3)
    S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=13, bulletIndent=2, spaceAfter=3.2)
    S["num"] = ParagraphStyle("num", parent=S["body"], leftIndent=17, bulletIndent=0, spaceAfter=3.5)
    S["cell"] = ParagraphStyle("cell", fontName=FONT, fontSize=8.2, leading=10.6, textColor=INK)
    S["cellb"] = ParagraphStyle("cellb", parent=S["cell"], fontName=SEMI)
    S["cellh"] = ParagraphStyle("cellh", parent=S["cell"], fontName=SEMI, textColor=colors.white)
    S["cellr"] = ParagraphStyle("cellr", parent=S["cell"], alignment=2)
    S["code"] = ParagraphStyle("code", fontName=MONO, fontSize=7.3, leading=9.3, textColor=INK)
    S["formula"] = ParagraphStyle("formula", fontName=FONT, fontSize=9.6, leading=14, textColor=INK, leftIndent=10)
    S["callt"] = ParagraphStyle("callt", fontName=SEMI, fontSize=8.6, leading=11, textColor=ACCENT, spaceAfter=2)
    S["callb"] = ParagraphStyle("callb", parent=S["body"], fontSize=9.1, leading=13, spaceAfter=0)
    S["tile_v"] = ParagraphStyle("tile_v", fontName=BOLD, fontSize=17, leading=20, textColor=INK)
    S["tile_l"] = ParagraphStyle("tile_l", fontName=FONT, fontSize=7.8, leading=10, textColor=MUTED)
    S["toc0"] = ParagraphStyle("toc0", fontName=SEMI, fontSize=10.5, leading=21, textColor=INK, leftIndent=0)
    S["toc1"] = ParagraphStyle("toc1", fontName=FONT, fontSize=8.8, leading=12.5, textColor=INK2, leftIndent=16)
    S["q"] = ParagraphStyle("q", fontName=SEMI, fontSize=9.4, leading=13, textColor=INK, spaceBefore=6, spaceAfter=2)
    S["a"] = ParagraphStyle("a", parent=S["body"], leftIndent=12, textColor=INK2)


_styles()
for _name in ("h1", "h2", "h3", "kicker", "q"):
    S[_name].keepWithNext = 1
for _name in ("body", "bullet", "num", "a", "lead", "callb"):
    S[_name].allowWidows = 0
    S[_name].allowOrphans = 0


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str], style: str = "bullet") -> list[Paragraph]:
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def numbered(items: list[str]) -> list[Paragraph]:
    return [Paragraph(t, S["num"], bulletText=f"{k}.") for k, t in enumerate(items, start=1)]


# ------------------------------------------------------------------ document
class Doc(BaseDocTemplate):
    def __init__(self, path: Path, title: str, footer: str, draft: bool):
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN + 4,
                         bottomMargin=MARGIN, title=title, author="Shane", subject=footer,
                         creator="docs/report/build_report.py")
        self.footer = footer
        self.draft = draft
        frame = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN - 4, id="body", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        cover = Frame(MARGIN, MARGIN, CONTENT_W, PAGE_H - 2 * MARGIN, id="cover", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("cover", [cover], onPage=self._cover_page),
                               PageTemplate("body", [frame], onPage=self._body_page)])
        self._h1 = 0

    def _cover_page(self, canv, doc) -> None:
        canv.saveState()
        canv.setFillColor(ACCENT)
        canv.rect(0, PAGE_H - 9 * mm, PAGE_W, 9 * mm, stroke=0, fill=1)
        if self.draft:
            self._draft_mark(canv)
        canv.restoreState()

    def _body_page(self, canv, doc) -> None:
        canv.saveState()
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.6)
        canv.line(MARGIN, MARGIN - 8, PAGE_W - MARGIN, MARGIN - 8)
        canv.setFont(FONT, 7.6)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN, MARGIN - 19, self.footer)
        canv.drawRightString(PAGE_W - MARGIN, MARGIN - 19, f"{doc.page}")
        canv.setFillColor(ACCENT)
        canv.rect(MARGIN, PAGE_H - MARGIN + 6, 22, 3, stroke=0, fill=1)
        if self.draft:
            self._draft_mark(canv)
        canv.restoreState()

    @staticmethod
    def _draft_mark(canv) -> None:
        canv.setFont(BOLD, 9)
        canv.setFillColor(OLD)
        canv.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 4, "DRAFT: final aligner and test results pending")

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name in ("h1", "h2"):
            level = 0 if flowable.style.name == "h1" else 1
            text = re.sub(r"<[^>]+>", "", flowable.getPlainText())
            key = f"s{self.seq.nextf('toc')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            if level == 0:
                self.notify("TOCEntry", (level, text, self.page, key))


def toc() -> TableOfContents:
    t = TableOfContents()
    t.levelStyles = [S["toc0"], S["toc1"]]
    t.dotsMinLevel = 0
    return t


def section(number: int, title: str, kicker: str | None = None) -> list:
    out = [Spacer(1, 14)]
    if kicker:
        out.append(P(kicker.upper(), "kicker"))
    out.append(Paragraph(f"{number}. {title}", S["h1"]))
    out.append(Spacer(1, 2))
    rule = Drawing(CONTENT_W, 4)
    rule.add(Rect(0, 1, 36, 2.4, fillColor=ACCENT, strokeColor=None))
    rule.add(Line(40, 2.2, CONTENT_W, 2.2, strokeColor=RULE, strokeWidth=0.6))
    out.append(rule)
    out.append(Spacer(1, 8))
    return out


def h2(text: str) -> Paragraph:
    return Paragraph(text, S["h2"])


def h3(text: str) -> Paragraph:
    return Paragraph(text, S["h3"])


# ------------------------------------------------------------------ components
def tiles(items: list[tuple[str, str]], cols: int = 3, fill=ACCENT_SOFT) -> Table:
    cells = [[Paragraph(v, S["tile_v"]), Paragraph(lbl, S["tile_l"])] for v, lbl in items]
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    while len(rows[-1]) < cols:
        rows[-1].append("")
    w = (CONTENT_W - (cols - 1) * 6) / cols
    data = [[_tile(c, w, fill) if c else "" for c in r] for r in rows]
    t = Table(data, colWidths=[w + (6 if k < cols - 1 else 0) for k in range(cols)], hAlign="LEFT")
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def _tile(cell, w, fill) -> Table:
    t = Table([[cell[0]], [cell[1]]], colWidths=[w])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (0, 0), 8),
                           ("BOTTOMPADDING", (0, -1), (-1, -1), 8), ("TOPPADDING", (0, 1), (-1, 1), 1),
                           ("BOTTOMPADDING", (0, 0), (0, 0), 0)]))
    return t


def callout(title: str, body: str | list, kind: str = "talk") -> Table:
    fills = {"talk": (ACCENT_SOFT, ACCENT), "note": (GRAY_SOFT, MUTED), "warn": (WARN_SOFT, WARN),
             "ok": (OK_SOFT, OK), "old": (OLD_SOFT, OLD)}
    fill, bar = fills[kind]
    st = ParagraphStyle("ct", parent=S["callt"], textColor=bar)
    content = [Paragraph(title, st)]
    if isinstance(body, str):
        content.append(Paragraph(body, S["callb"]))
    else:
        content.extend(Paragraph(b, ParagraphStyle("cb", parent=S["callb"], leftIndent=10, bulletIndent=0,
                                                   spaceAfter=2), bulletText="•") for b in body)
    t = Table([[content]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fill), ("LINEBEFORE", (0, 0), (0, -1), 2.6, bar),
                           ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return KeepTogether([t, Spacer(1, 8)])


def table(header: list[str], rows: list[list], widths: list[float] | None = None, num_cols: tuple = (),
          bold_first: bool = False, highlight_rows: tuple = ()) -> Table:
    n = len(header)
    widths = widths or [CONTENT_W / n] * n
    scale = CONTENT_W / sum(widths)
    widths = [w * scale for w in widths]

    def cell(v, c, r):
        if isinstance(v, (Paragraph, Image, Table, Drawing)):
            return v
        style = "cellr" if c in num_cols else ("cellb" if (bold_first and c == 0) or r in highlight_rows else "cell")
        if r in highlight_rows and c in num_cols:
            style = "cellr"
            return Paragraph(f"<b>{v}</b>", S[style])
        return Paragraph(str(v), S[style])

    data = [[Paragraph(h, ParagraphStyle("hh", parent=S["cellh"], alignment=2 if k in num_cols else 0))
             for k, h in enumerate(header)]]
    data += [[cell(v, c, r) for c, v in enumerate(row)] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    st = [("BACKGROUND", (0, 0), (-1, 0), INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 4.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
          ("LINEBELOW", (0, 1), (-1, -1), 0.4, RULE)]
    for r in range(1, len(data)):
        if r % 2 == 0:
            st.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    for r in highlight_rows:
        st.append(("BACKGROUND", (0, r + 1), (-1, r + 1), ACCENT_SOFT))
    t.setStyle(TableStyle(st))
    return t


def image(path: Path, width: float = CONTENT_W, caption: str | None = None, keep: bool = True) -> list:
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width, height=width * ih / iw)
    img.hAlign = "LEFT"
    out = [img]
    if caption:
        out.append(P(caption, "caption"))
    return [KeepTogether(out)] if keep else out


def side_by_side(left: list, right: list, gap: float = 10) -> Table:
    w = (CONTENT_W - gap) / 2
    t = Table([[left, right]], colWidths=[w + gap, w])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (0, 0), gap), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    return t


def half_image(path: Path, caption: str | None = None) -> list:
    return image(path, (CONTENT_W - 10) / 2, caption, keep=False)


def snippet(rel: str, start: str, n: int, title: str | None = None, max_w: int = 104) -> list:
    """A real excerpt of the project's source: n lines from the first line containing `start`."""
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    k = next((i for i, ln in enumerate(lines) if start in ln), None)
    if k is None:
        return [P(f"(excerpt not found: {rel})", "small")]
    body = textwrap.dedent("\n".join(lines[k:k + n]))
    body = "\n".join(ln if len(ln) <= max_w else ln[:max_w - 1] + "…" for ln in body.splitlines())
    pre = Preformatted(body, S["code"])
    label = P(f"<font name='{MONO}'>{rel}</font>  (line {k + 1})" + (f" · {title}" if title else ""), "small")
    t = Table([[pre]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CODE_BG), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LINEBEFORE", (0, 0), (0, -1), 2, RULE)]))
    return [KeepTogether([label, t, Spacer(1, 8)])]


def formula(lines: list[str]) -> KeepTogether:
    t = Table([[[Paragraph(x, S["formula"]) for x in lines]]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.white), ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    return KeepTogether([t, Spacer(1, 8)])


# ------------------------------------------------------------------ diagrams
def _arrow(d: Drawing, x1, y1, x2, y2, color=MUTED, width=1.0) -> None:
    import math
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    ang = math.atan2(y2 - y1, x2 - x1)
    s = 4.2
    p1 = (x2 - s * math.cos(ang - 0.45), y2 - s * math.sin(ang - 0.45))
    p2 = (x2 - s * math.cos(ang + 0.45), y2 - s * math.sin(ang + 0.45))
    d.add(Polygon([x2, y2, *p1, *p2], fillColor=color, strokeColor=color, strokeWidth=0.4))


def _box(d: Drawing, x, y, w, h, title, sub=None, kind="ours") -> None:
    fills = {"ours": (ACCENT_SOFT, ACCENT), "old": (OLD_SOFT, OLD), "rule": (GRAY_SOFT, MUTED),
             "io": (colors.white, RULE)}
    fill, stroke = fills[kind]
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.9))
    ty = y + h / 2 + (3.2 if sub else -2.6)
    d.add(String(x + w / 2, ty, title, fontName=SEMI, fontSize=7.4, fillColor=INK, textAnchor="middle"))
    if sub:
        d.add(String(x + w / 2, y + h / 2 - 7.2, sub, fontName=FONT, fontSize=6.4, fillColor=MUTED,
                     textAnchor="middle"))


def pipeline_diagram() -> Drawing:
    W, H = CONTENT_W, 172
    d = Drawing(W, H)
    bw, bh, gap = 84, 34, 14
    row1 = [("Video / audio", "ffmpeg -> 16 kHz", "io"), ("Whisper ASR", "PRETRAINED (kept)", "old"),
            ("Sync + alignment", "RANSAC + NW alignment", "rule"), ("Evidence fusion", "our 4-gram KN LM", "ours"),
            ("Punctuation", "our BiLSTM tagger", "ours")]
    row2 = [("Corrector", "our Transformer + BPE", "ours"), ("Spelling + rules", "SymSpell, LanguageTool", "rule"),
            ("Word timing", "our CTC aligner", "ours"), ("Cue splitting", "DP + our POS tagger", "ours"),
            (".srt / .vtt", "+ change log, diff", "io")]
    x0 = (W - (5 * bw + 4 * gap)) / 2
    y1, y2 = H - 52, 40
    for k, (t, s, kind) in enumerate(row1):
        x = x0 + k * (bw + gap)
        _box(d, x, y1, bw, bh, t, s, kind)
        if k:
            _arrow(d, x - gap + 1, y1 + bh / 2, x - 1, y1 + bh / 2)
    # wrap from the end of row 1 to the start of row 2 (row 2 runs left to right)
    xr = x0 + 4 * (bw + gap) + bw / 2
    d.add(Line(xr, y1, xr, y1 - 12, strokeColor=MUTED, strokeWidth=1))
    d.add(Line(xr, y1 - 12, x0 + bw / 2, y1 - 12, strokeColor=MUTED, strokeWidth=1))
    _arrow(d, x0 + bw / 2, y1 - 12, x0 + bw / 2, y2 + bh + 1)
    order = [row2[1], row2[0], row2[2], row2[3], row2[4]]   # spelling/rules run before the corrector
    for k, (t, s, kind) in enumerate(order):
        x = x0 + k * (bw + gap)
        _box(d, x, y2, bw, bh, t, s, kind)
        if k:
            _arrow(d, x - gap + 1, y2 + bh / 2, x - 1, y2 + bh / 2)
    # legend
    ly = 10
    items = [("ours", "Trained from scratch in this project"), ("old", "Pretrained (Whisper only)"),
             ("rule", "Algorithm / rules, no learned weights"), ("io", "Input / output")]
    lx = x0
    for kind, label in items:
        fills = {"ours": (ACCENT_SOFT, ACCENT), "old": (OLD_SOFT, OLD), "rule": (GRAY_SOFT, MUTED),
                 "io": (colors.white, RULE)}
        f, s = fills[kind]
        d.add(Rect(lx, ly, 9, 9, rx=2, ry=2, fillColor=f, strokeColor=s, strokeWidth=0.8))
        d.add(String(lx + 13, ly + 1.8, label, fontName=FONT, fontSize=6.8, fillColor=INK2))
        lx += 13 + pdfmetrics.stringWidth(label, FONT, 6.8) + 16
    return d


def stack_diagram(columns: list[list[tuple[str, str | None, str]]], trunk: list[tuple[str, str | None, str]],
                  height: float | None = None, box_h: float = 28, col_w: float = 128) -> Drawing:
    """Parallel input columns (top to bottom) that merge into a shared trunk below them."""
    gap_y = 12
    n_col = max(len(c) for c in columns)
    H = height or (n_col + len(trunk)) * (box_h + gap_y) + 8
    W = CONTENT_W
    d = Drawing(W, H)
    ncols = len(columns)
    gx = 12
    total = ncols * col_w + (ncols - 1) * gx
    x0 = (W - total) / 2
    top = H - box_h - 2
    bottoms = []
    for c, col in enumerate(columns):
        x = x0 + c * (col_w + gx)
        y = top
        for k, (t, s, kind) in enumerate(col):
            _box(d, x, y, col_w, box_h, t, s, kind)
            if k < len(col) - 1:
                _arrow(d, x + col_w / 2, y - 1, x + col_w / 2, y - gap_y + 1)
            y -= box_h + gap_y
        bottoms.append((x + col_w / 2, y + box_h + gap_y))
    tw = min(W * 0.6, max(col_w * 1.6, 220))
    tx = (W - tw) / 2
    ty = top - n_col * (box_h + gap_y)
    for bx, by in bottoms:
        _arrow(d, bx, by - 1, min(max(bx, tx + 10), tx + tw - 10), ty + box_h + 1)
    for k, (t, s, kind) in enumerate(trunk):
        _box(d, tx, ty, tw, box_h, t, s, kind)
        if k < len(trunk) - 1:
            _arrow(d, tx + tw / 2, ty - 1, tx + tw / 2, ty - gap_y + 1)
        ty -= box_h + gap_y
    return d


def chain_diagram(items: list[tuple[str, str | None, str]], labels: list[str] | None = None, box_w: float = 66,
                  box_h: float = 34) -> Drawing:
    """A left-to-right chain of boxes with optional labels on the arrows."""
    n = len(items)
    gap = (CONTENT_W - n * box_w) / max(n - 1, 1)
    H = box_h + 26
    d = Drawing(CONTENT_W, H)
    y = 6
    for k, (t, s, kind) in enumerate(items):
        x = k * (box_w + gap)
        _box(d, x, y, box_w, box_h, t, s, kind)
        if k:
            _arrow(d, x - gap + 2, y + box_h / 2, x - 2, y + box_h / 2)
            if labels and labels[k - 1]:
                d.add(String(x - gap / 2, y + box_h / 2 + 5, labels[k - 1], fontName=FONT, fontSize=6.4,
                             fillColor=MUTED, textAnchor="middle"))
    return d
