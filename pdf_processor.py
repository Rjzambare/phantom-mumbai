from PyPDF2 import PdfReader

def extract_text_pypdf2(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
