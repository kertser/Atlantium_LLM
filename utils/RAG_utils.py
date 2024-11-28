# Document parsing and RAG-related functions
import fitz  # PyMuPDF for PDFs
fitz.TOOLS.mupdf_display_errors(False)  # Suppress MuPDF errors

import docx  # for Word documents
import openpyxl  # for Excel files
from docx import Document
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from image_store import ImageStore
from config import CONFIG


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

            # Check both vertical and horizontal proximity
            if abs(block_center_y - image_center_y) < context_range and \
                    abs(block_center_x - image_center_x) < context_range * 2:  # Wider horizontal range
                text = block[4].strip()
                if text:
                    nearby_text.append(text)

        return " ".join(nearby_text)
    except Exception as e:
        print(f"Error extracting text context: {e}")
        return ""


def get_relevant_images(query_context: str, image_store: ImageStore, threshold: float = 0.3):
    """Get images relevant to the query with improved matching"""
    relevant_images = []
    query_terms = set(query_context.lower().split())

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
            print(f"Error processing image {img_id}: {e}")
            continue

    # Sort by relevance
    relevant_images.sort(key=lambda x: x['similarity'], reverse=True)
    return relevant_images[:5]  # Return top 5 most relevant images

def extract_text_and_images_from_pdf(pdf_path):
    """Extracts text and images with their context from a PDF file."""
    text = ""
    image_data = []  # List of tuples (image, context, page_num)
    min_size = 50

    try:
        pdf_document = fitz.open(pdf_path)

        for page_num in range(pdf_document.page_count):
            try:
                page = pdf_document.load_page(page_num)
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"

                # Get images and their locations
                image_list = page.get_images(full=True)
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
                                print(f"Skipping small image ({image.width}x{image.height}) on page {page_num}")
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

                            print(f"Extracted image: {image.width}x{image.height} from page {page_num}")
                            print(f"Context: {context[:100]}...")  # Print first 100 chars of context

                            image_data.append({
                                'image': image,
                                'context': context,
                                'page_num': page_num,
                                'bbox': [img_bbox.x0, img_bbox.y0, img_bbox.x1, img_bbox.y1]
                            })

                    except Exception as e:
                        print(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue

            except Exception as e:
                print(f"Error processing page {page_num}: {e}")

        pdf_document.close()

    except Exception as e:
        print(f"Error opening or processing PDF {pdf_path}: {e}")

    return text, image_data

def extract_text_and_images_from_word(doc_path):
    """Extract text and images from a Word document."""
    try:
        doc = Document(doc_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        images = []

        # Extract images
        for rel in doc.part.rels.values():
            try:
                if "image" in rel.target_ref:
                    image_data = rel.target_part.blob
                    image = Image.open(BytesIO(image_data))
                    images.append(image)
            except Exception as e:
                print(f"Error extracting an image from {doc_path}: {e}")

        return text, images
    except Exception as e:
        print(f"Error processing Word document {doc_path}: {e}")
        return "", []

def extract_text_and_images_from_excel(excel_path):
    """Extract text and images from an Excel file."""
    try:
        workbook = openpyxl.load_workbook(excel_path)
        text = ""
        images = []

        for sheet in workbook.worksheets:
            # Extract text
            try:
                for row in sheet.iter_rows(values_only=True):
                    text += " ".join([str(cell) for cell in row if cell]) + "\n"
            except Exception as e:
                print(f"Error processing text in sheet {sheet.title}: {e}")

            # Extract images
            try:
                for image in sheet._images:
                    if hasattr(image, '_data'):
                        img_data = image._data()
                        img = Image.open(BytesIO(img_data))
                        images.append(img)
            except Exception as e:
                print(f"Error processing images in sheet {sheet.title}: {e}")

        return text, images
    except Exception as e:
        print(f"Error opening Excel file {excel_path}: {e}")
        return "", []

def chunk_text(text, chunk_size=CONFIG.CHUNK_SIZE, overlap=CONFIG.CHUNK_OVERLAP):
    """Split text into chunks with overlap to maintain context."""
    if not text or chunk_size < CONFIG.MIN_CHUNK_SIZE:
        return []

    words = text.split()
    chunks = []
    start_idx = 0

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

        # Create chunk
        chunk = ' '.join(words[start_idx:end_idx])
        chunks.append(chunk)

        # Move start index, accounting for overlap
        start_idx = end_idx - overlap if end_idx < len(words) else end_idx

    return chunks