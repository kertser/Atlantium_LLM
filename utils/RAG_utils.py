# Document parsing and RAG-related functions

import fitz  # PyMuPDF for PDFs

fitz.TOOLS.mupdf_display_errors(False)  # Suppress MuPDF errors

import docx  # for Word documents
import openpyxl  # for Excel files
from docx import Document
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from io import BytesIO


def extract_text_and_images_from_word(docx_path, image_store):
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        image_ids = []

        # Extract images
        for rel in doc.part.rels.values():
            try:
                if "image" in rel.target_ref:
                    image_data = rel.target_part.blob
                    image = Image.open(BytesIO(image_data))

                    # Store image and get ID
                    image_id = image_store.store_image(
                        image=image,
                        source_doc=str(docx_path),
                        page_num=0  # Word docs don't have pages, use 0
                    )
                    image_ids.append(image_id)
            except Exception as e:
                print(f"Error extracting an image from {docx_path}: {e}")

        return text, image_ids
    except Exception as e:
        print(f"Error processing Word document {docx_path}: {e}")
        return "", []


def extract_text_and_images_from_pdf(pdf_path, image_store):
    """Extracts text and images from a PDF file."""
    text = ""
    image_ids = []

    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)

        for page_num in range(pdf_document.page_count):
            try:
                # Extract text from the page
                page = pdf_document.load_page(page_num)
                page_text = page.get_text("text")
                if page_text:
                    text += page_text + "\n"

                # Extract images from the page
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]  # Image reference number
                        base_image = pdf_document.extract_image(xref)

                        # Ensure base_image is valid
                        if not base_image or "image" not in base_image:
                            continue

                        image_bytes = base_image["image"]

                        # Convert to PIL Image
                        image = Image.open(BytesIO(image_bytes))

                        # Store image and get ID
                        image_id = image_store.store_image(
                            image=image,
                            source_doc=str(pdf_path),
                            page_num=page_num
                        )
                        image_ids.append(image_id)

                    except UnidentifiedImageError:
                        print(f"Skipping image {img_index} on page {page_num}: Unidentified image error.")
                    except Exception as e:
                        if "cmsOpenProfileFromMem" in str(e):
                            print(f"Skipping image {img_index} on page {page_num}: Invalid color profile.")
                        else:
                            print(f"Error extracting image {img_index} on page {page_num}: {e}")
                        continue

            except Exception as e:
                print(f"Error processing page {page_num}: {e}")

        pdf_document.close()

    except Exception as e:
        print(f"Error opening or processing PDF {pdf_path}: {e}")

    return text, image_ids


def extract_text_and_images_from_excel(excel_path, image_store):
    try:
        workbook = openpyxl.load_workbook(excel_path)
        text = ""
        image_ids = []

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

                        # Store image and get ID
                        image_id = image_store.store_image(
                            image=img,
                            source_doc=str(excel_path),
                            page_num=workbook.worksheets.index(sheet)  # Use sheet index as page number
                        )
                        image_ids.append(image_id)
            except Exception as e:
                print(f"Error processing images in sheet {sheet.title}: {e}")

        return text, image_ids
    except Exception as e:
        print(f"Error opening Excel file {excel_path}: {e}")
        return "", []


def chunk_text(text, chunk_size=512, overlap=100):
    """
    Split text into chunks with overlap to maintain context
    """
    if not text or chunk_size <= 0:
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