import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List

from config import CONFIG
from utils.img_utils import ImageProcessor

# import openpyxl  # for Excel files
from openpyxl.reader.excel import load_workbook
import pymupdf  # PyMuPDF for PDFs
from PIL import Image, UnidentifiedImageError
from docx import Document

import re
import nltk
# Download NLTK data (if not already downloaded)
nltk.download('punkt')

# Suppress MuPDF warnings
logging.getLogger("fitz").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

pymupdf.TOOLS.mupdf_display_errors(False)


def extract_image_safely(page, xref: int, base_image: Dict) -> Optional[Image.Image]:
    """Safely extract and convert image with multiple fallback methods."""
    image_bytes = base_image.get("image")

    if not image_bytes:
        return None

    methods = [
        lambda: Image.open(BytesIO(image_bytes)),
        lambda: Image.frombytes(
            "RGB",
            (page.get_pixmap().width, page.get_pixmap().height),
            page.get_pixmap(
                matrix=pymupdf.Matrix(1, 1),
                colorspace="rgb",
                clip=page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
            ).samples
        ),
        lambda: Image.frombytes(
            "RGB",
            (page.get_pixmap().width, page.get_pixmap().height),
            page.get_pixmap(colorspace="rgb").samples
        )
    ]

    last_error = None
    for method in methods:
        try:
            image = method()
            if image and ImageProcessor.validate_image_data(image):
                return image
        except Exception as e:
            last_error = e
            continue

    logging.debug(f"All image extraction methods failed for xref {xref}: {last_error}")
    return None

def extract_text_around_image(page, image_bbox, context_range=CONFIG.MAX_CONTEXT_RANGE):
    """Extract text around an image's location on the page."""
    try:
        blocks = page.get_text("blocks")
        image_center_y = (image_bbox[1] + image_bbox[3]) / 2
        image_center_x = (image_bbox[0] + image_bbox[2]) / 2

        nearby_text = []
        for block in blocks:
            block_center_y = (block[1] + block[3]) / 2
            block_center_x = (block[0] + block[2]) / 2

            if abs(block_center_y - image_center_y) < context_range and \
                    abs(block_center_x - image_center_x) < context_range * 2:
                text = block[4].strip()
                if text:
                    nearby_text.append(text)

        return " ".join(nearby_text)
    except Exception as e:
        logger.error(f"Error extracting text context: {e}")
        return ""

def extract_text_and_images_from_pdf(pdf_path):
    """Extracts text and images with their context from a PDF file."""
    text = ""
    image_data = []
    pdf_document = None

    try:
        pdf_document = pymupdf.open(pdf_path)
        doc_name = Path(pdf_path).name
        logger.info(f"Processing PDF document: {doc_name}")

        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document.load_page(page_num)
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"

                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)

                        if not base_image or "image" not in base_image:
                            continue

                        try:
                            image_bytes = base_image["image"]
                            image = Image.open(BytesIO(image_bytes))

                            # Force load to verify image is valid
                            image.load()

                            # Get dimensions and calculate aspect ratio
                            width, height = image.size
                            aspect_ratio = max(width / height, height / width)

                            # Filter out small images and icons using CONFIG settings
                            if (width < CONFIG.MIN_IMAGE_SIZE or
                                    height < CONFIG.MIN_IMAGE_SIZE or
                                    max(width, height) < CONFIG.MIN_ICON_SIZE or
                                    aspect_ratio > CONFIG.MAX_ASPECT_RATIO):
                                logger.debug(
                                    f"Skipping small/icon image on page {page_num + 1}: "
                                    f"{width}x{height} pixels, aspect ratio: {aspect_ratio:.2f}"
                                )
                                continue

                            # Extract context with CONFIG.MAX_CONTEXT_RANGE
                            context = ""
                            for img_bbox in page.get_image_rects(xref):
                                context = extract_text_around_image(
                                    page,
                                    img_bbox,
                                    context_range=CONFIG.MAX_CONTEXT_RANGE
                                )
                                break

                            # Handle color modes while preserving quality
                            if image.mode in ('RGB', 'RGBA'):
                                pass  # Keep as is
                            elif image.mode == 'CMYK':
                                image = image.convert('RGB')
                            elif image.mode == 'P':
                                if 'transparency' in image.info:
                                    image = image.convert('RGBA')
                                else:
                                    image = image.convert('RGB')
                            elif image.mode in ('L', 'LA'):
                                if image.mode == 'LA':
                                    image = image.convert('RGBA')
                            else:
                                image = image.convert('RGB')

                            # Set image DPI if not already set
                            if 'dpi' not in image.info:
                                image.info['dpi'] = CONFIG.IMAGE_DPI

                            image_data.append({
                                'image': image,
                                'context': context,
                                'page_num': page_num + 1,
                                'caption': f"Image {img_index + 1} from {doc_name} (Page {page_num + 1})",
                                'dimensions': f"{width}x{height}",
                                'dpi': CONFIG.IMAGE_DPI,
                                'bits': CONFIG.IMAGE_BITS,
                                'original_mode': image.mode
                            })

                            logger.info(
                                f"Processed image {img_index + 1} from page {page_num + 1}: "
                                f"{width}x{height} pixels, mode={image.mode}"
                            )

                        except UnidentifiedImageError:
                            logger.debug(
                                f"Skipping unidentifiable image on page {page_num + 1}"
                            )
                            continue
                        except Exception as e:
                            logger.debug(
                                f"Error processing image on page {page_num + 1}: {str(e)}"
                            )
                            continue

                    except Exception as e:
                        logger.error(
                            f"Error extracting image on page {page_num + 1}: {str(e)}"
                        )
                        continue

            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
    finally:
        if pdf_document is not None:
            pdf_document.close()

    return text, image_data


def extract_text_and_images_from_word(doc_path):
    """
    Extract text and images from a Word document with enhanced image processing.

    Args:
        doc_path: Path to the Word document

    Returns:
        tuple: (extracted_text, list of image_data dictionaries)
        Each image_data dictionary contains:
            - image: PIL Image object
            - context: Text context around the image (currently empty)
            - page_num: Page number (always 1 for Word docs)
            - caption: Image caption
    """
    try:
        doc = Document(doc_path)
        doc_name = Path(doc_path).name
        logger.info(f"Processing Word document: {doc_name}")

        text = "\n".join([para.text for para in doc.paragraphs])
        images_data = []

        # Extract images from relationships
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    image = Image.open(BytesIO(image_data))

                    if image.mode in CONFIG.VALID_IMAGE_MODES:
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        background.paste(image, mask=image.split()[-1])
                        image = background
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')

                    if image.width < CONFIG.MIN_IMAGE_SIZE or image.height < CONFIG.MIN_IMAGE_SIZE:
                        logger.info(f"Skipping small image ({image.width}x{image.height}) in {doc_name}")
                        continue

                    # Currently, surrounding_text is empty.
                    surrounding_text = ""
                    img_data = {
                        'image': image,
                        'context': surrounding_text,
                        'page_num': 1,  # Word docs don't have pages in the same way as PDFs
                        'caption': f"Image from {doc_name}",
                        'dimensions': f"{image.width}x{image.height}",
                        'format': image.format,
                        'mode': 'RGB'
                    }
                    images_data.append(img_data)
                    logger.info(f"Processed image ({img_data['dimensions']}) from {doc_name}")

                except UnidentifiedImageError as uie:
                    logger.error(f"Invalid or corrupted image in {doc_name}: {uie}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing image from {doc_name}: {e}")
                    continue

        logger.info(f"Completed processing {doc_name}: extracted {len(images_data)} valid images")
        return text, images_data

    except Exception as e:
        logger.error(f"Error processing Word document {doc_path}: {e}", exc_info=True)
        return "", []


def extract_text_and_images_from_excel(excel_path):
    """
    Extract text and images from an Excel file.

    Args:
        excel_path (str or Path): Path to the Excel file

    Returns:
        tuple: (extracted_text, list of image_data dictionaries)
    """

    workbook = None

    try:
        workbook = load_workbook(excel_path, data_only=True)
        doc_name = Path(excel_path).name
        logger.info(f"Processing Excel document: {doc_name}")

        text = ""
        images = []

        for sheet_name in workbook.sheetnames:
            try:
                worksheet = workbook[sheet_name]

                # Extract text from cells
                for row in worksheet.iter_rows(values_only=True):
                    row_text = " ".join([str(cell) if cell else "" for cell in row]).strip()
                    if row_text:
                        text += row_text + "\n"

                # Extract images
                for image in getattr(worksheet, '_images', []):
                    try:
                        img_content = getattr(image, 'ref', None)
                        if not img_content or not hasattr(img_content.image, 'content'):
                            continue

                        img = Image.open(BytesIO(img_content.image.content))
                        width, height = img.size

                        # Skip small images
                        if width < CONFIG.MIN_IMAGE_SIZE or height < CONFIG.MIN_IMAGE_SIZE:
                            logger.debug(f"Skipping small image ({width}x{height}) in {sheet_name}")
                            continue

                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        anchor = getattr(image, 'anchor', None)
                        cell_ref = str(anchor) if anchor else "Unknown"

                        # Extract context
                        context = "No context available"
                        if anchor and hasattr(anchor, '_from'):
                            try:
                                row, col = anchor._from.row, anchor._from.col
                                context = " ".join(
                                    str(worksheet.cell(r, c).value or "").strip()
                                    for r in range(max(1, row - 1), row + 2)
                                    for c in range(max(1, col - 1), col + 2)
                                )
                            except Exception:
                                logger.debug(f"Error extracting context for image in {sheet_name}")

                        images.append({
                            'image': img,
                            'context': context.strip(),
                            'page_num': 1,
                            'caption': f"Image from {doc_name} - Sheet: {sheet_name} (Cell: {cell_ref})",
                            'dimensions': f"{width}x{height}",
                            'sheet': sheet_name,
                            'cell_reference': cell_ref,
                            'source_document': str(excel_path)
                        })
                        logger.info(f"Processed image ({width}x{height}) in {sheet_name}")

                    except UnidentifiedImageError:
                        logger.debug(f"Unidentified image format in {sheet_name}")
                    except Exception as e:
                        logger.error(f"Error processing image in {sheet_name}: {e}")

            except Exception as e:
                logger.error(f"Error processing sheet {sheet_name}: {e}")

        logger.info(f"Completed processing {doc_name}: {len(images)} images extracted")
        return text, images

    except Exception as e:
        logger.error(f"Error processing Excel file {excel_path}: {e}")
        return "", []

    finally:
        if workbook:
            try:
                workbook.close()
            except Exception as e:
                logger.debug(f"Error closing workbook: {e}")

def chunk_text(text: str, source_path: str,
               chunk_size: int = CONFIG.CHUNK_SIZE,
               overlap: int = CONFIG.CHUNK_OVERLAP) -> List[Dict]:
    """
    Splits text into chunks with overlap and enhanced metadata,
    filtering out non-meaningful content (headers, footers, TOC, etc.).

    Args:
        text: The text to be chunked (e.g., extracted from PDF/Word/Excel).
        source_path: Path to the source document.
        chunk_size: Number of words in each chunk (default from CONFIG).
        overlap: Number of words overlapped between consecutive chunks.

    Returns:
        List of dictionaries, each containing:
         - 'text': the chunk text
         - 'metadata': relevant information such as start_idx, end_idx, etc.
    """
    # -------------------------------------------------------------------------
    # 0. Quick exit if no text or invalid chunk_size
    # -------------------------------------------------------------------------
    if not text or chunk_size < CONFIG.MIN_CHUNK_SIZE:
        return []

    # -------------------------------------------------------------------------
    # 1. Compile Regex Patterns
    # -------------------------------------------------------------------------
    # Header-like patterns
    header_patterns = [
        r'^\s*(?:header|heading)\s*$',  # e.g., "header" alone
        r'^\s*(?:page\s+\d+|p\.\s*\d+)\s*$',  # e.g., "page 12" alone
        r'^\s*(?:chapter|section)\s+\d+\s*$'  # e.g., "chapter 2"
    ]

    # Footer-like patterns
    footer_patterns = [
        r'^\s*(?:footer)\s*$',
        r'^\s*copyright\s+©?\s*\d{4}',
        r'^\s*all\s+rights\s+reserved\s*$',
    ]

    # Table-of-contents patterns
    toc_patterns = [
        r'^\s*(?:table\s+of\s+contents|contents)\s*$',  # "table of contents"
        r'^.*?\.{3,}.*?\d+(?:-\d+)?$',  # e.g., "Topic ... 12" or "Topic ... 12-34"
        r'(?:\.{3,}\s*\d+(?:-\d+)?){2,}',  # multiple segments of "... 6-22 ... 6-23"
    ]

    # Disclaimers
    disclaimer_patterns = [
        r'^\s*disclaimer:\s*$',
        r'^\s*(?:confidential|proprietary)\s+document\s*$'
    ]

    # Combine everything into one set of filters
    all_filters = header_patterns + footer_patterns + toc_patterns + disclaimer_patterns
    compiled_filters = [re.compile(p, re.IGNORECASE) for p in all_filters]

    # -------------------------------------------------------------------------
    # 2. Text Cleaning
    # -------------------------------------------------------------------------
    def clean_text(input_text: str) -> str:
        """Remove HTML, extra newlines, repeated spaces, and unwanted symbols."""
        if not input_text:
            return ""
        # Remove any HTML tags
        text_no_html = re.sub(r'<.*?>', '', input_text)
        # Collapse 3+ newlines to double newlines
        text_no_extra_newlines = re.sub(r'\n{3,}', '\n\n', text_no_html)
        # Collapse multiple spaces
        text_single_spaced = re.sub(r'\s+', ' ', text_no_extra_newlines)
        # Whitelist: keep letters, digits, underscores, certain punctuation
        text_clean = re.sub(r'[^-\w\s.,!?;:()\'"]', '', text_single_spaced)
        return text_clean.strip()

    text = clean_text(text)

    # -------------------------------------------------------------------------
    # 3. Line-Based Filtering
    #    - If line is suspicious, remove it before forming paragraphs
    #    - We also track line repetition to remove repeated short lines
    # -------------------------------------------------------------------------
    line_freq = {}

    def is_filter_line(line: str) -> bool:
        """
        Returns True if the line should be filtered out
        (e.g., matches known header/footer/TOC patterns, is repeated too often, etc.)
        """
        line_stripped = line.strip()
        if not line_stripped:
            return True  # blank lines -> skip

        # Frequency-based approach for short lines
        # (like repeated "Chapter 2" at top/bottom of pages)
        short_alnum = re.sub(r'[\W\d_]+', '', line_stripped)  # remove digits & symbols
        if len(short_alnum) < 3:
            # This line is mostly digits/punctuation or extremely short
            line_freq[line_stripped] = line_freq.get(line_stripped, 0) + 1
            # Filter if it repeats 3+ times
            if line_freq[line_stripped] >= 3:
                return True

        # Check compiled patterns (headers, footers, disclaimers, etc.)
        for pattern in compiled_filters:
            # .search() instead of .match() so we find patterns anywhere in line
            if pattern.search(line_stripped):
                return True

        # Additional check: if line has 2+ "dot+pageNumber" combos,
        # or if it has multiple "..." clusters, it's likely a TOC line
        if re.search(r'(?:\.{3,}\s*\d+(?:-\d+)?){2,}', line_stripped):
            return True

        # If line has 3+ sets of "..." => likely a TOC or junk
        if len(re.findall(r'\.{3,}', line_stripped)) >= 3:
            return True

        # Otherwise, it's okay
        return False

    # Split text by newline and filter line-by-line
    raw_lines = text.split('\n')
    filtered_lines = []
    for ln in raw_lines:
        if not is_filter_line(ln):
            filtered_lines.append(ln)

    # -------------------------------------------------------------------------
    # 4. Reconstruct Paragraphs After Filtering
    # -------------------------------------------------------------------------
    paragraphs = []
    paragraph_buffer = []
    for line in filtered_lines:
        stripped = line.strip()
        if stripped:
            paragraph_buffer.append(stripped)
        else:
            # A blank line ends a paragraph
            if paragraph_buffer:
                paragraphs.append(' '.join(paragraph_buffer))
                paragraph_buffer = []
    # Add last paragraph if we ended with text
    if paragraph_buffer:
        paragraphs.append(' '.join(paragraph_buffer))

    # -------------------------------------------------------------------------
    # 5. Paragraph-Level Filtering (Optional)
    #    - If a paragraph is too short or mostly dots, skip it
    # -------------------------------------------------------------------------
    def is_meaningful_paragraph(p: str) -> bool:
        # Remove non-alphanumerics for length check
        p_alnum = re.sub(r'[\W_]+', '', p)
        if len(p_alnum) < CONFIG.MIN_CHUNK_SIZE:
            return False

        # Check for too many "..." lines inside the paragraph
        dot_lines = sum(1 for line in p.split('\n') if '...' in line)
        if dot_lines > 1:
            return False

        return True

    meaningful_paragraphs = [p for p in paragraphs if is_meaningful_paragraph(p)]

    # -------------------------------------------------------------------------
    # 6. Sentence Splitting with NLTK
    # -------------------------------------------------------------------------
    sentences = []
    try:
        for para in meaningful_paragraphs:
            # Use nltk's sentence tokenizer for better accuracy
            para_sents = nltk.sent_tokenize(para)
            # Clean each sentence (optional)
            para_sents = [clean_text(s) for s in para_sents if s.strip()]
            sentences.extend(para_sents)
    except Exception:
        # Fallback: naive split if nltk is unavailable
        for para in meaningful_paragraphs:
            for sent in re.split(r'[.!?]+', para):
                cleaned = clean_text(sent)
                if cleaned:
                    sentences.append(cleaned)

    # -------------------------------------------------------------------------
    # 7. Build Overlapping Chunks
    # -------------------------------------------------------------------------
    words = []
    for sentence in sentences:
        words.extend(sentence.split())

    chunks = []
    start_idx = 0
    chunk_number = 0
    sentence_endings = {'.', '!', '?', ':', ';'}

    while start_idx < len(words):
        end_idx = start_idx + chunk_size
        if end_idx >= len(words):
            end_idx = len(words)
        else:
            # Look for a natural sentence break going backward
            # within the overlap region
            natural_break = None
            search_start = max(start_idx, end_idx - overlap)
            for i in range(end_idx - 1, search_start - 1, -1):
                if any(words[i].endswith(se) for se in sentence_endings):
                    natural_break = i + 1
                    break
            if natural_break:
                end_idx = natural_break

        chunk_text = ' '.join(words[start_idx:end_idx]).strip()

        # Check if this chunk is meaningful
        alnum_len = len(re.sub(r'[\W_]+', '', chunk_text))
        if alnum_len >= CONFIG.MIN_CHUNK_SIZE:
            chunk = {
                'text': chunk_text,
                'metadata': {
                    'source_path': source_path,
                    'chunk_number': chunk_number,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'relative_path': str(Path(source_path).relative_to(CONFIG.RAW_DOCUMENTS_PATH)),
                    'chunk_length': len(chunk_text),
                    'word_count': len(chunk_text.split())
                }
            }
            chunks.append(chunk)
            chunk_number += 1

        if end_idx >= len(words):
            break
        # Overlap
        start_idx = end_idx - overlap

    return chunks
