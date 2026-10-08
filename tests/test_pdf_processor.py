"""Tests for Phase 1 - PDF text extraction."""
from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from src.pdf_processor import extract_text_from_pdf


def make_text_pdf(pages: list[str]) -> bytes:
    """Build an in-memory PDF with one string per page (test fixture)."""
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        if text:
            c.drawString(100, 750, text)
        c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


def test_single_page_extraction():
    data = make_text_pdf(["Hello RAG world"])
    result = extract_text_from_pdf(data)
    assert len(result) == 1
    assert result[0]["page"] == 1
    assert "Hello RAG world" in result[0]["text"]


def test_page_numbers_preserved_when_middle_page_empty():
    data = make_text_pdf(["page one content", "", "page three content"])
    result = extract_text_from_pdf(BytesIO(data))
    assert [r["page"] for r in result] == [1, 3]
    assert "page one" in result[0]["text"]
    assert "page three" in result[1]["text"]
    assert len(result) == 2


def test_path_and_bytes_inputs_agree(tmp_path):
    data = make_text_pdf(["path vs bytes check"])
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(data)
    from_path = extract_text_from_pdf(pdf_path)
    from_bytes = extract_text_from_pdf(data)
    assert from_path == from_bytes
    assert from_path[0]["page"] == 1


def test_invalid_file_raises_value_error():
    with pytest.raises(ValueError, match="Invalid file"):
        extract_text_from_pdf(b"this is not a pdf")


def test_empty_bytes_raises_value_error():
    with pytest.raises(ValueError, match="Empty file"):
        extract_text_from_pdf(b"")


def test_scanned_pdf_with_no_text_returns_empty():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    buf = BytesIO()
    writer.write(buf)
    result = extract_text_from_pdf(buf.getvalue())
    assert result == []


def test_streamlit_like_upload_object():
    """Streamlit UploadedFile exposes .read()/.getvalue()/.name."""
    data = make_text_pdf(["upload simulation"])

    class FakeUpload:
        def __init__(self, payload: bytes):
            self._buf = BytesIO(payload)
            self.name = "fake.pdf"

        def read(self, *a, **k):
            return self._buf.read(*a, **k)

        def seek(self, *a, **k):
            return self._buf.seek(*a, **k)

        def getvalue(self):
            return self._buf.getvalue()

    result = extract_text_from_pdf(FakeUpload(data))
    assert len(result) == 1
    assert "upload simulation" in result[0]["text"]
