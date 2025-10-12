from PyPDF2 import PdfReader

def read_pdf_to_text(pdf_path):
    """
    Reads a PDF file and extracts text from each page.

    Args:
        pdf_path (str): Path to the input PDF file.

    Returns:
        list of str: A list where each element is the text of a page.
    """
    reader = PdfReader(pdf_path)
    pages_text = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        page_text = f"--- Page {i} ---\n{text.strip() if text else ''}\n"
        pages_text.append(page_text)

    return pages_text

def write_text_to_file(text_list, output_txt):
    """
    Writes a list of text strings to a text file.

    Args:
        text_list (list of str): List of text strings to write.
        output_txt (str): Path to the output text file.
    """
    with open(output_txt, "w", encoding="utf-8") as f:
        for page_text in text_list:
            f.write(page_text + "\n")

    print(f"✅ All pages processed. Text saved to '{output_txt}'")

# Example usage
if __name__ == "__main__":
    input_pdf = r"C:\RAG\budget_speech_2025_26.pdf"
    output_txt = r"C:\RAG\2025_2026.txt"

    extracted_text = read_pdf_to_text(input_pdf)
    write_text_to_file(extracted_text, output_txt)