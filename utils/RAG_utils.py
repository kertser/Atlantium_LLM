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


def chunk_text(text: str, source_path: str, chunk_size: int = CONFIG.CHUNK_SIZE,
               overlap: int = CONFIG.CHUNK_OVERLAP) -> List[Dict]:
    """
    Split text into chunks with overlap and enhanced metadata, filtering out non-meaningful content.

    Args:
        text: The text to be chunked
        source_path: Path to the source document
        chunk_size: Size of each chunk (default: from CONFIG)
        overlap: Number of words to overlap between chunks (default: from CONFIG)

    Returns:
        List of dictionaries containing chunks and their metadata
    """
    if not text or chunk_size < CONFIG.MIN_CHUNK_SIZE:
        return []

    # Define patterns for filtering out non-meaningful content with more precise matching
    header_patterns = [
        r'^\s*(?:header|heading)\s*$',  # Only exact matches
        r'^\s*(?:page\s+\d+|p\.\s*\d+)\s*$',  # Only standalone page numbers
        r'^\s*(?:chapter|section)\s+\d+\s*$'  # Only standalone chapter/section markers
    ]

    footer_patterns = [
        r'^\s*(?:footer)\s*$',  # Only exact matches
        r'^\s*copyright\s+©?\s*\d{4}',  # More specific copyright pattern
        r'^\s*all\s+rights\s+reserved\s*$',  # Exact "all rights reserved"
        r'^\s*\d+\s*$'  # Only standalone page numbers
    ]

    toc_patterns = [
        r'^\s*(?:table\s+of\s+contents|contents)\s*$',  # Only exact TOC headers
        r'^\s*(?:\d+\.){1,3}\s+[A-Za-z].*?\s+\d+\s*$'  # More specific TOC entry pattern
    ]

    disclaimer_patterns = [
        r'^\s*disclaimer:\s*$',  # Only standalone disclaimer headers
        r'^\s*(?:confidential|proprietary)\s+document\s*$'  # Only specific confidentiality headers
    ]

    # Compile all patterns
    patterns = [re.compile(p, re.IGNORECASE) for p in
                header_patterns + footer_patterns + toc_patterns + disclaimer_patterns]

    # Clean and preprocess text
    def clean_text(input_text: str) -> str:
        if not input_text:
            return ""
        # Remove HTML tags
        input_text = re.sub(r'<.*?>', '', input_text)
        # Remove multiple newlines but preserve paragraph breaks
        input_text = re.sub(r'\n{3,}', '\n\n', input_text)
        # Remove multiple spaces
        input_text = re.sub(r'\s+', ' ', input_text)
        # Preserve important punctuation while removing other special characters
        input_text = re.sub(r'[^\w\s.!?,:;()\'\-\"]', '', input_text)
        return input_text.strip()

    def is_meaningful_content(content_text: str) -> bool:
        # Skip empty or whitespace-only content
        if not content_text.strip():
            return False

        # Only apply patterns to short lines (likely headers/footers)
        if len(content_text.split()) < 10:
            if any(pattern.match(content_text) for pattern in patterns):
                return False

        # Skip if too short
        if len(content_text.strip()) < CONFIG.MIN_CHUNK_SIZE:
            return False

        # Skip if mostly special characters or numbers
        text_clean = re.sub(r'[\W\d]', '', content_text)
        if len(text_clean) < len(content_text) * 0.2:  # Relaxed to 20% letters
            return False

        return True

    # Clean the text
    text = clean_text(text)

    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Filter out non-meaningful paragraphs
    meaningful_paragraphs = [p for p in paragraphs if is_meaningful_content(p)]

    # Define sentence ending characters
    sentence_endings = {'.', '!', '?', ':', ';'}  # Added more sentence separators

    # Use NLTK sentence tokenizer for better sentence boundary detection
    try:
        sentences = []
        for para in meaningful_paragraphs:
            sentences.extend(nltk.sent_tokenize(para))
    except Exception as e:
        # Fallback to simple sentence splitting if NLTK fails
        sentences = []
        for para in meaningful_paragraphs:
            for sent in re.split(r'[.!?]+', para):
                if sent.strip():
                    sentences.append(sent.strip())

    # Process meaningful sentences into chunks
    words = []
    for sentence in sentences:
        words.extend(sentence.split())

    chunks = []
    start_idx = 0
    chunk_number = 0

    while start_idx < len(words):
        end_idx = start_idx + chunk_size
        if end_idx < len(words):
            chunk_breakpoint = end_idx

            # Look for natural sentence endings within the overlap region
            for i in range(max(start_idx + chunk_size - overlap, start_idx), end_idx):
                word = words[i]
                if any(word.endswith(end) for end in sentence_endings):
                    # Found a natural break point
                    chunk_breakpoint = i + 1
                    break
            end_idx = chunk_breakpoint

        chunktext = ' '.join(words[start_idx:end_idx])

        # Only add chunk if it contains meaningful content
        if is_meaningful_content(chunktext):
            chunk = {
                'text': chunktext,
                'metadata': {
                    'source_path': source_path,
                    'chunk_number': chunk_number,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'relative_path': str(Path(source_path).relative_to(CONFIG.RAW_DOCUMENTS_PATH)),
                    'is_paragraph_start': start_idx == 0 or words[start_idx - 1].endswith('\n'),
                    'is_paragraph_end': end_idx >= len(words) or words[end_idx - 1].endswith('\n'),
                    'chunk_length': len(chunktext),
                    'word_count': len(chunktext.split())
                }
            }
            chunks.append(chunk)
            chunk_number += 1

        start_idx = end_idx - overlap if end_idx < len(words) else end_idx

    return chunks
