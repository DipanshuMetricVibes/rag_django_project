import os
import json
from google import genai

# ------------------ CONFIG ------------------
PROJECT_ID = "metricvibes-1718777660306"
REGION = "us-central1"

INPUT_IMAGE_DIR = "report-images"
OUTPUT_TEXT_DIR = "text-extract"
os.makedirs(OUTPUT_TEXT_DIR, exist_ok=True)

# ------------------ Init Gemini Client ------------------
client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

# ------------------ Gemini Prompt Template ------------------
def build_prompt(image_data):
    return [{
        "role": "user",
        "parts": [
            {
                "text": """You are a visual analysis expert.
Analyze this image of a document page. Extract the content and represent it as complete, clean, structured text.

🔹 If it has:
- **Text**: extract exactly as it is.
- **Tables**: recreate with proper data and format in readable plain text.
- **Charts/Graphs**: convert to detailed captions with data points and meaning.
- **Diagrams/Images**: describe in words with complete context.
- 📢 Don't miss anything.
- ✅ Final output should feel like someone just read that page fully.

Format the response as plain readable text."""
            },
            {"inline_data": {"mime_type": "image/png", "data": image_data}},
        ]
    }]

# ------------------ Main Function ------------------
def process_images_to_text():
    grouped = {}

    for filename in sorted(os.listdir(INPUT_IMAGE_DIR)):
        if filename.lower().endswith(".png"):
            base_name = "_".join(filename.split("_")[:-1])  # e.g., "report1_page1" → "report1"
            grouped.setdefault(base_name, []).append(filename)

    for base_file, image_list in grouped.items():
        print(f"\n📄 Processing PDF: {base_file}")
        full_text = ""

        for img_file in sorted(image_list):
            image_path = os.path.join(INPUT_IMAGE_DIR, img_file)
            with open(image_path, "rb") as f:
                image_data = f.read()

            contents = build_prompt(image_data)

            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=contents
                )
                text = response.text.strip()
                full_text += f"\n--- Page: {img_file} ---\n{text}\n\n"
                print(f"✅ Done: {img_file}")
            except Exception as e:
                print(f"❌ Error on {img_file}: {e}")

        # Save to output file
        output_path = os.path.join(OUTPUT_TEXT_DIR, f"{base_file}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"📄 Saved full extracted content to: {output_path}")

# ------------------ Run ------------------
if __name__ == "__main__":
    process_images_to_text()
