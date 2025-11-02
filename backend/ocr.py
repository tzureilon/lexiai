import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a given file (PDF or image).
    """
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path)
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page)
        return text
    else:
        return pytesseract.image_to_string(Image.open(file_path))
