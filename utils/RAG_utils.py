# Document parsing and RAG-related functions
import logging
import pymupdf  # PyMuPDF for PDFs
import openpyxl  # for Excel files
from docx import Document
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from utils.image_store import ImageStore
from config import CONFIG
from pathlib import Path

# Configure logging at module level
logger = logging.getLogger(__name__)

def extract_text_around_image(page, image_bbox, context_range=100):
    """Extract text around an image's location on the page with improved context"""
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
    """Get images relevant to the query with improved matching"""
    relevant_images = []
    query_terms = set(query_context.lower().split())

    if not query_terms:
        logger.warning("Empty query terms, cannot calculate relevance")
        return []

    for img_id, metadata in image_store.metadata.items():
        try:
            # Get all text associated with the image
            context = metadata.get("context", "").lower()
            caption = metadata.get("caption", "").lower()
            source = metadata.get("source_document", "").lower()

            # Split text into terms
            context_terms = set(context.split())
            caption_terms = set(caption.split())
            source_terms = set(source.split())

            # Calculate term overlap
            term_overlap = len(query_terms & (context_terms | caption_terms | source_terms))
            if term_overlap > 0:
                score = term_overlap / len(query_terms)
                if score >= threshold:
                    base64_img = image_store.get_base64_image(img_id)
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
    return relevant_images[:5]  # 5 most relevant


def extract_text_and_images_from_pdf(pdf_path):
    """Extracts text and images with their context from a PDF file."""
    text = ""
    image_data = []
    min_size = 50
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

                # Get images and their locations
                image_list = page.get_images(full=True)
                logger.info(f"Found {len(image_list)} images on page {page_num + 1}")

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)

                        if not base_image or "image" not in base_image:
                            continue

                        # Get image location on page
                        for img_bbox in page.get_image_rects(xref):
                            context = extract_text_around_image(page, img_bbox)

                            image_bytes = base_image["image"]
                            image = Image.open(BytesIO(image_bytes))

                            if image.width < min_size or image.height < min_size:
                                logger.info(
                                    f"Skipping small image ({image.width}x{image.height}) on page {page_num + 1}")
                                continue

                            # Convert to RGB if needed
                            if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                                background = Image.new('RGB', image.size, (255, 255, 255))
                                if image.mode == 'P':
                                    image = image.convert('RGBA')
                                background.paste(image, mask=image.split()[-1])
                                image = background
                            elif image.mode != 'RGB':
                                image = image.convert('RGB')

                            image_data.append({
                                'image': image,
                                'context': context,
                                'page_num': page_num + 1,
                                'caption': f"Image {img_index + 1} from {doc_name} (Page {page_num + 1})",
                                'bbox': [img_bbox.x0, img_bbox.y0, img_bbox.x1, img_bbox.y1]
                            })
                            logger.info(f"Processed image {img_index + 1} from page {page_num + 1}")

                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue

        logger.info(f"Completed processing {doc_name}: {len(image_data)} images extracted")

    except Exception as e:
        logger.error(f"Error opening or processing PDF {pdf_path}: {e}")
    finally:
        if pdf_document is not None:
            try:
                pdf_document.close()
            except Exception as e:
                logger.error(f"Error closing PDF document: {e}")

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
            - context: Text context around the image
            - page_num: Page number (always 1 for Word docs)
            - caption: Image caption
    """
    min_size = 50

    try:
        doc = Document(doc_path)
        doc_name = Path(doc_path).name
        logger.info(f"Processing Word document: {doc_name}")

        # Extract all text from paragraphs
        text = "\n".join([para.text for para in doc.paragraphs])
        images_data = []

        # Extract images from relationships
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    # Extract image data
                    image_data = rel.target_part.blob
                    image = Image.open(BytesIO(image_data))

                    # Handle image mode conversion
                    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
                        # Create white background for transparent images
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        background.paste(image, mask=image.split()[-1])
                        image = background
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Verify image dimensions and quality
                    if image.width < min_size or image.height < min_size:
                        logger.info(f"Skipping small image ({image.width}x{image.height}) in {doc_name}")
                        continue

                    # Try to find text near the image (could be enhanced based on document structure)
                    surrounding_text = ""

                    # Create image data dictionary with enhanced metadata
                    img_data = {
                        'image': image,
                        'context': surrounding_text,
                        'page_num': 1,  # Word docs don't have native page numbers
                        'caption': f"Image from {doc_name}",
                        'dimensions': f"{image.width}x{image.height}",
                        'format': image.format,
                        'mode': 'RGB'  # We ensure all images are in RGB mode
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
    """Extract text and images from an Excel file."""
    try:
        workbook = openpyxl.load_workbook(excel_path)
        doc_name = Path(excel_path).name
        logger.info(f"Processing Excel document: {doc_name}")

        text = ""
        images = []

        for sheet in workbook.worksheets:
            # Extract text
            for row in sheet.iter_rows(values_only=True):
                text += " ".join([str(cell) if cell is not None else "" for cell in row]) + "\n"

            # Extract images
            for image in sheet._images:
                try:
                    if hasattr(image, '_data'):
                        img_data = image._data()
                        img = Image.open(BytesIO(img_data))

                        # Convert to RGB if needed
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


def chunk_text(text: str, source_path: str, chunk_size=CONFIG.CHUNK_SIZE, overlap=CONFIG.CHUNK_OVERLAP):
    """Split text into chunks with overlap and enhanced metadata."""
    if not text or chunk_size < CONFIG.MIN_CHUNK_SIZE:
        return []

    words = text.split()
    chunks = []
    start_idx = 0
    chunk_number = 0

    while start_idx < len(words):
        # Calculate end index for current chunk
        end_idx = start_idx + chunk_size

        # If we're not at the end of the text, try to find a good breakpoint
        if end_idx < len(words):
            # Look for the last period or newline in the overlap region
            breakpoint = end_idx
            for i in range(max(start_idx + chunk_size - overlap, start_idx), end_idx):
                if words[i].endswith('.') or words[i].endswith('\n'):
                    breakpoint = i + 1
                    break
            end_idx = breakpoint

        # Create chunk with metadata
        chunk = {
            'text': ' '.join(words[start_idx:end_idx]),
            'metadata': {
                'source_path': source_path,
                'chunk_number': chunk_number,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'relative_path': str(Path(source_path).relative_to(CONFIG.RAW_DOCUMENTS_PATH))
            }
        }
        chunks.append(chunk)
        chunk_number += 1

        # Move start index, accounting for overlap
        start_idx = end_idx - overlap if end_idx < len(words) else end_idx

    return chunks
