# Document parsing and RAG-related functions

import fitz  # PyMuPDF for PDFs
fitz.TOOLS.mupdf_display_errors(False)  # Suppress MuPDF errors

import docx  # for Word documents
import openpyxl  # for Excel files
from docx import Document

from PIL import Image, UnidentifiedImageError
from io import BytesIO

def extract_text_and_images_from_word(docx_path):
    try:
        doc = Document(docx_path)
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
                print(f"Error extracting an image from {docx_path}: {e}")

        return text, images
    except Exception as e:
        print(f"Error processing Word document {docx_path}: {e}")
        return "", []

def extract_text_and_images_from_pdf(pdf_path):
    """Extracts text and images from a PDF file."""
    text = ""
    images = []

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
                        images.append(image)

                    except UnidentifiedImageError:
                        # Log and skip if the image cannot be identified
                        print(f"Skipping image {img_index} on page {page_num}: Unidentified image error.")
                    except Exception as e:
                        if "cmsOpenProfileFromMem" in str(e):
                            # Suppress color profile errors
                            print(f"Skipping image {img_index} on page {page_num}: Invalid color profile.")
                        else:
                            # Log other errors but continue
                            print(f"Error extracting image {img_index} on page {page_num}: {e}")
                        continue

            except Exception as e:
                print(f"Error processing page {page_num}: {e}")

        pdf_document.close()

    except Exception as e:
        print(f"Error opening or processing PDF {pdf_path}: {e}")

    return text, images


def extract_text_and_images_from_excel(excel_path):
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
                    if isinstance(image, OpenPyXLImage):
                        img_data = image._data()  # Get the raw image data
                        img = Image.open(BytesIO(img_data))
                        images.append(img)
            except Exception as e:
                print(f"Error processing images in sheet {sheet.title}: {e}")

        return text, images
    except Exception as e:
        print(f"Error opening Excel file {excel_path}: {e}")
        return "", []

#%%
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