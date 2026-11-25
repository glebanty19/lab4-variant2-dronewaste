"""Small shared helpers for the LR4 and LR6 Word reports."""

from __future__ import annotations

import math
from typing import Any

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Pt


CODE_URL = "https://github.com/glebanty19/lab4-variant2-dronewaste"


def num(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.3f}".replace(".", ",")


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "D9EAF7")
        cell._tc.get_or_add_tcPr().append(shade)
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.name = "Arial"
            run.font.size = Pt(9)
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = str(value)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(9)
    doc.add_paragraph()


def add_code_link(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    paragraph.add_run("Код ЛР4 на GitHub: ")
    relationship = doc.part.relate_to(CODE_URL, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.append(underline)
    run.append(properties)
    text = OxmlElement("w:t")
    text.text = CODE_URL
    run.append(text)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
    paragraph.paragraph_format.space_after = Pt(10)
