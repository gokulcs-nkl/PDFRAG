"""PDF text extraction (Phase 1).

Problem solved:
    Convert unstructured PDF binaries into structured, page-aligned text
    that downstream stages (chunking -> embedding -> FAISS) can consume.

Input:
    A PDF as a file path, raw bytes, or a binary file-like object
    (e.g. Streamlit's UploadedFile).

Output:
    A list of dicts, one per non-empty page:
        {"text": "...", "page": 1}   # page is 1-indexed, original numbering

Why pypdf:
    Pure-Python, lightweight, sufficient for text-based PDFs. PyMuPDF is
    faster/more accurate on complex layouts, but pypdf was an explicit
    project choice for simpler install. The function boundary keeps the
    library swappable (only this module imports pypdf).

Connection to next component:
    `chunker.py` (Phase 2) takes this page list and splits each page's
    text into overlapping chunks, adding document name + chunk_id.

Empty-page policy:
    Pages whose extracted text is None/whitespace-only are skipped, but
    surviving pages keep their ORIGINAL 1-indexed page numbers so that
    source citations ("example.pdf - Page 7") stay correct.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Union

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Public record shape. Kept as a plain dict (not a dataclass) so the
# return value matches the spec example verbatim and serialises trivially.
PageText = Dict[str, object]  # {"text": str, "page": int}

PdfSource = Union[str, Path, bytes, BinaryIO]


def _to_bytes(source: PdfSource) -> bytes:
    """Normalise any accepted input to raw PDF bytes."""
    if isinstance(source, bytes):
        data = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found: {path}")
        data = path.read_bytes()
    elif hasattr(source, "getvalue"):
        # Streamlit UploadedFile and BytesIO both expose getvalue().
        try:
            data = source.getvalue()  # type: ignore[union-attr]
        except Exception:
            data = source.read()  # type: ignore[union-attr]
    elif hasattr(source, "read"):
        try:
            source.seek(0)  # type: ignore[union-attr]
        except Exception:
            pass
        data = source.read()  # type: ignore[union-attr]
    else:
        raise ValueError(
            f"Unsupported PDF source type: {type(source).__name__}. "
            "Expected a file path, bytes, or binary file-like object."
        )

    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(
            "Could not read PDF bytes from the provided source. "
            "Expected a PDF file path, bytes, or binary file-like object."
        )
    data = bytes(data)
    if len(data) == 0:
        raise ValueError("Empty file: no PDF bytes received.")
    if not data.lstrip().startswith(b"%PDF-"):
        raise ValueError(
            "Invalid file: missing PDF header (%PDF-). "
            "Only valid .pdf files are accepted."
        )
    return data


def extract_text_from_pdf(
    source: PdfSource,
    password: str | None = None,
) -> List[PageText]:
    """Extract text page by page from a PDF.

    Args:
        source: File path, raw bytes, or binary file-like object.
        password: Optional password for encrypted PDFs. An empty-string
            password is tried automatically when none is supplied.

    Returns:
        List of {"text": str, "page": int} for each page containing
        extractable text. Empty (scanned/image-only) pages are skipped.
        Returns [] when the PDF has no extractable text at all.

    Raises:
        FileNotFoundError: Path does not exist.
        ValueError: Invalid/empty bytes, corrupt PDF, or wrong password.
    """
    data = _to_bytes(source)

    try:
        reader = PdfReader(BytesIO(data))
    except (PdfReadError, Exception) as exc:
        raise ValueError(
            "Invalid or corrupt PDF file: could not be parsed as a PDF."
        ) from exc

    # Encrypted PDFs: try user password, then empty password, else fail
    # with a user-friendly (non-leaking) error.
    if getattr(reader, "is_encrypted", False):
        tried = []
        for candidate in ([password] if password else []) + [""]:
            try:
                if reader.decrypt(candidate or ""):
                    break
                tried.append(repr(candidate))
            except Exception:
                continue
        else:
            # decrypt() returns 0 on failure for all candidates
            if getattr(reader, "is_encrypted", False):
                raise ValueError(
                    "Encrypted PDF: a password is required to read this file."
                )

    pages: List[PageText] = []
    try:
        num_pages = len(reader.pages)
    except Exception as exc:
        raise ValueError("Invalid or corrupt PDF file: cannot read pages.") from exc

    for i in range(num_pages):
        try:
            raw = reader.pages[i].extract_text()
        except Exception:
            # One bad page must not kill the whole document; skip it.
            continue
        text = (raw or "").strip()
        if not text:
            continue  # empty / scanned page: skip gracefully
        pages.append({"text": text, "page": i + 1})  # 1-indexed

    return pages
