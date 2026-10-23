"""
OCR for uploaded patient reports, using EasyOCR (pure pip install, no
separate system binary needed -- unlike Tesseract, which requires a
platform-specific install outside of pip and is a common source of
Windows setup failures).

Scope, stated honestly: this module EXTRACTS and DISPLAYS the text in a
report, and flags which of our 4 known disease-class keywords appear in
it. It does NOT interpret, diagnose, or reason about the report's
clinical meaning -- that would be a much harder, unvalidated NLP task
and is out of scope here.
"""

import re
import threading

_reader = None
_reader_lock = threading.Lock()


def _get_reader():
    """Lazily initializes the EasyOCR reader once (it's slow to load)."""
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                import easyocr
                _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


# Keywords tied to this project's 4 known disease classes, plus a few
# common synonyms that tend to appear in real radiology report language.
KEYWORD_MAP = {
    "COVID": ["covid", "covid-19", "sars-cov-2", "coronavirus"],
    "Normal": ["normal", "no acute findings", "unremarkable", "clear lungs"],
    "Lung_Opacity": ["opacity", "opacities", "consolidation", "infiltrate"],
    "Viral_Pneumonia": ["pneumonia", "viral pneumonia", "bronchopneumonia"],
}


def extract_text(image_path):
    """Returns the raw OCR'd text from an image file as a single string."""
    reader = _get_reader()
    results = reader.readtext(image_path, detail=0)
    return " ".join(results)


def find_keywords(text):
    """Returns a list of disease-class names whose keywords were found in
    the OCR'd text (case-insensitive substring match)."""
    text_lower = text.lower()
    found = []
    for class_name, keywords in KEYWORD_MAP.items():
        if any(kw in text_lower for kw in keywords):
            found.append(class_name)
    return found
