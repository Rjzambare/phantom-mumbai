from pdfminer.high_level import extract_text

def extract_text_pdfminer(file_path):
    try:
        text = extract_text(file_path)
        return text.strip()  # Remove leading/trailing whitespace
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
