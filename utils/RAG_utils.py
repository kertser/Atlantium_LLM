import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List

import openpyxl  # for Excel files
import pymupdf  # PyMuPDF for PDFs
from PIL import Image, UnidentifiedImageError
from docx import Document

import re
import nltk
# Download NLTK data (if not already downloaded)
nltk.download('punkt')

from config import CONFIG
from utils.img_utils import ImageStore, ImageProcessor

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


def get_relevant_images(query_context: str, image_store: ImageStore, threshold: float = 0.3):
    """
    Get images relevant to the query with improved matching.

    Args:
        query_context: A string containing the query or context to match.
        image_store: An ImageStore instance for getting images and metadata.
        threshold: Minimum overlap-to-query-terms ratio for relevance.

    Returns:
        A list of dictionaries, each containing image ID, base64 data, caption,
        context, and similarity score.
    """
    relevant_images = []
    query_terms = set(query_context.lower().split())

    if not query_terms:
        logger.warning("Empty query terms, cannot calculate relevance")
        return []

    for img_id, metadata in image_store.metadata.items():
        try:
            context = metadata.get("context", "").lower()
            caption = metadata.get("caption", "").lower()
            source = metadata.get("source_document", "").lower()

            context_terms = set(context.split())
            caption_terms = set(caption.split())
            source_terms = set(source.split())

            term_overlap = len(query_terms & (context_terms | caption_terms | source_terms))
            if term_overlap > 0:
                score = term_overlap / len(query_terms)
                if score >= threshold:
                    base64_img = image_store.get_base64(img_id)
                    if base64_img:
                        relevant_images.append({
                            "id": img_id,
                            "base64": base64_img,
                            "caption": metadata.get("caption", "No caption available"),
                            "context": metadata.get("context", ""),
                            "similarity": score
                        })
        except Exception as e:
            logger.error(f"Error processing image {img_id}: {e}")
            continue

    relevant_images.sort(key=lambda x: x['similarity'], reverse=True)
    return relevant_images[:5]  # Return up to 5 most relevant images


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
    """
    try:
        workbook = openpyxl.load_workbook(excel_path, read_only=True)
        doc_name = Path(excel_path).name
        logger.info(f"Processing Excel document: {doc_name}")

        text = ""
        images = []

        for sheet in workbook.worksheets:
            # Extract text from cells
            for row in sheet.iter_rows(values_only=True):
                text += " ".join([str(cell) if cell is not None else "" for cell in row]) + "\n"

            # Extract images
            for image in sheet._images:
                try:
                    if hasattr(image, '_data'):
                        img_data = image._data()
                        img = Image.open(BytesIO(img_data))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        images.append({
                            'image': img,
                            'context': '',
                            'page_num': 1,
                            'caption': f"Image from {doc_name} - Sheet: {sheet.title}"
                        })
                        logger.info(f"Processed image from sheet {sheet.title}")
                except Exception as e:
                    logger.error(f"Error processing image in Excel sheet {sheet.title}: {e}")
                    continue

        logger.info(f"Completed processing {doc_name}: {len(images)} images extracted")
        return text, images

    except Exception as e:
        logger.error(f"Error processing Excel file {excel_path}: {e}")
        return "", []


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

    # Define patterns for filtering out non-meaningful content
    header_patterns = [
        r'^.*(?:header|heading).*$',
        r'^\s*(?:page\s+\d+|p\.\s*\d+)\s*$',
        r'^\s*(?:chapter|section)\s+\d+.*$'
    ]
    footer_patterns = [
        r'^.*(?:footer|copyright|all rights reserved).*$',
        r'^\s*\d+\s*$'  # Page numbers
    ]
    toc_patterns = [
        r'^(?:table of contents|contents|toc).*$',
        r'^\s*(?:\d+\.)+\s+.*\s+\d+\s*$',  # TOC entries with page numbers
    ]
    disclaimer_patterns = [
        r'^.*(?:disclaimer|legal).*$',
        r'^.*(?:confidential|proprietary).*$'
    ]

    # Compile all patterns
    patterns = [re.compile(p, re.IGNORECASE) for p in header_patterns + footer_patterns + toc_patterns + disclaimer_patterns]

    # Clean and preprocess text
    def clean_text(text: str) -> str:
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters and excessive punctuation
        text = re.sub(r'[^\w\s.!?]', '', text)
        return text.strip()

    def is_meaningful_content(text: str) -> bool:
        # Skip if matches any header/footer/TOC/disclaimer patterns
        if any(pattern.match(text) for pattern in patterns):
            return False

        # Skip if too short
        if len(text.strip()) < CONFIG.MIN_CHUNK_SIZE:
            return False

        # Skip if mostly special characters or numbers
        text_clean = re.sub(r'[\W\d]', '', text)
        if len(text_clean) < len(text) * 0.3:  # Less than 30% letters
            return False

        return True

    # Clean the text
    text = clean_text(text)

    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    # Filter out non-meaningful paragraphs
    meaningful_paragraphs = [p for p in paragraphs if is_meaningful_content(p)]

    # Define sentence ending characters
    sentence_endings = {'.', '!', '?'}

    # Use NLTK sentence tokenizer for better sentence boundary detection
    sentences = []
    for para in meaningful_paragraphs:
        sentences.extend(nltk.sent_tokenize(para))

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
            breakpoint = end_idx

            # Look for natural sentence endings within the overlap region
            for i in range(max(start_idx + chunk_size - overlap, start_idx), end_idx):
                word = words[i]
                if any(word.endswith(end) for end in sentence_endings):
                    # Found a natural break point
                    breakpoint = i + 1
                    break
            end_idx = breakpoint

        chunk_text = ' '.join(words[start_idx:end_idx])

        # Only add chunk if it contains meaningful content
        if is_meaningful_content(chunk_text):
            chunk = {
                'text': chunk_text,
                'metadata': {
                    'source_path': source_path,
                    'chunk_number': chunk_number,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'relative_path': str(Path(source_path).relative_to(CONFIG.RAW_DOCUMENTS_PATH)),
                    'is_paragraph_start': start_idx == 0 or words[start_idx - 1].endswith('\n'),
                    'is_paragraph_end': end_idx >= len(words) or words[end_idx - 1].endswith('\n')
                }
            }
            chunks.append(chunk)
            chunk_number += 1

        start_idx = end_idx - overlap if end_idx < len(words) else end_idx

    return chunks
