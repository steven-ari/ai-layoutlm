from pathlib import Path

import pdfminer
from pdfminer.high_level import extract_pages

from layoutlm_utils.bbox import normalize_bbox, bbox_bottom_left_to_top_left, bbox_split_on_paragraph


def filter_for_valid_paragraph(page: pdfminer.layout.LTPage) -> pdfminer.layout.LTTextBoxHorizontal:
    for element in page:
        if isinstance(element, pdfminer.layout.LTTextBoxHorizontal) and element.get_text().strip() not in ['', ' ']:
            yield element


def extract_metadata_from_pdf(pdf_path: Path) -> dict:
    # reference: https://stackoverflow.com/a/69151177
    pages = extract_pages(pdf_path)
    pages_data = []

    for page in pages:
        page_data = []
        for paragraph in filter_for_valid_paragraph(page):
            bbox_paragraph = (paragraph.x0, paragraph.y0, paragraph.x1, paragraph.y1)
            bbox_paragraph_normalized = normalize_bbox(bbox_paragraph, page.height, page.width)
            bbox_paragraph = bbox_bottom_left_to_top_left(bbox_paragraph_normalized)
            text_paragraph = paragraph.get_text()
            words_n_bbox_paragraph = bbox_split_on_paragraph(text_paragraph, bbox_paragraph)
            page_data.extend(words_n_bbox_paragraph)
        pages_data.append((page_data, (page.height, page.width)))

    return create_metadata_from_pages_data(pages_data)


def create_metadata_from_pages_data(
        pages_data: list[tuple[list[tuple[str, tuple[int, int, int, int]]], tuple[int, int]]]
) -> dict:
    pages_metadata = []
    for page, page_shape in pages_data:
        page_text = " ".join(word for word, _ in page)
        pages_metadata.append({"page_data": page, "page_text": page_text, "page_shape": page_shape})

    document_text = " ".join(page["page_text"] for page in pages_metadata)
    document_data = {"document_text": document_text}

    metadata = {"document_data": document_data, "pages_data": pages_metadata}

    return metadata
