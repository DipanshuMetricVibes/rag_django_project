import os
import fitz  # PyMuPDF

# ----- Config -----
PDF_DIR = "reports"
OUTPUT_IMAGE_DIR = "report-images"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

def convert_pdf_to_images():
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            pdf_name = os.path.splitext(filename)[0]

            print(f"\n📄 Processing {filename}...")

            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=200)
                    image_filename = f"{pdf_name}_page{page_num+1}.png"
                    image_path = os.path.join(OUTPUT_IMAGE_DIR, image_filename)

                    pix.save(image_path)
                    print(f"🖼️ Saved {image_filename}")
                doc.close()
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
    
    print(f"\n✅ Done! All PDFs converted to images in '{OUTPUT_IMAGE_DIR}'.")

if __name__ == "__main__":
    convert_pdf_to_images()
