
from pathlib import Path
from collections import Counter
import json
import shutil

import pymupdf
from llama_index.core import SimpleDirectoryReader


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "maintenance_machine.pdf"
DOCUMENT_NAME = PDF_PATH.stem

DOCUMENT_DIR = BASE_DIR / "data" / "documents" / DOCUMENT_NAME
ORIGINAL_DIR = DOCUMENT_DIR / "original"
TEXT_DIR = DOCUMENT_DIR / "text"
TABLES_DIR = DOCUMENT_DIR / "tables"
IMAGES_DIR = DOCUMENT_DIR / "images"
PROCESSED_DIR = DOCUMENT_DIR / "processed"

for directory in [
    ORIGINAL_DIR,
    TEXT_DIR,
    TABLES_DIR,
    IMAGES_DIR,
    PROCESSED_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Conserver une copie du PDF original
original_pdf = ORIGINAL_DIR / PDF_PATH.name

if not original_pdf.exists():
    shutil.copy2(PDF_PATH, original_pdf)


# ==================================================
# 2. EXTRACTION DU TEXTE
# ==================================================

print("1. Extraction du texte...")

documents = SimpleDirectoryReader(
    input_files=[str(PDF_PATH)]
).load_data()

text_elements = []

for document in documents:
    page_number = int(document.metadata["page_label"])
    content = document.text.strip()

    if not content:
        continue

    element = {
        "type": "text",
        "content": content,
        "metadata": {
            "source": PDF_PATH.name,
            "page": page_number,
        },
    }

    text_elements.append(element)

    # Un fichier JSON par page
    output_file = TEXT_DIR / f"page_{page_number:03d}.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(element, file, ensure_ascii=False, indent=2)


# ==================================================
# 3. EXTRACTION DES TABLEAUX ET DES IMAGES
# ==================================================

print("2. Extraction des tableaux et des images...")

pdf = pymupdf.open(PDF_PATH)

table_elements = []
image_elements = []

total_images = 0

try:
    for page_number, page in enumerate(pdf, start=1):

        # ---------- TABLEAUX ----------
        tables = page.find_tables()

        for table_number, table in enumerate(
            tables.tables, start=1
        ):
            rows = table.extract()

            table_text = "\n".join(
                " | ".join(
                    str(cell).strip() if cell is not None else ""
                    for cell in row
                )
                for row in rows
            )

            if not table_text.strip():
                continue

            element = {
                "type": "table",
                "content": table_text,
                "metadata": {
                    "source": PDF_PATH.name,
                    "page": page_number,
                    "table_number": table_number,
                },
            }

            table_elements.append(element)

            output_file = (
                TABLES_DIR
                / f"page_{page_number:03d}_table_{table_number:02d}.json"
            )

            with output_file.open("w", encoding="utf-8") as file:
                json.dump(element, file, ensure_ascii=False, indent=2)

        # ---------- IMAGES ----------
        for image_number, image_info in enumerate(
            page.get_images(full=True), start=1
        ):
            xref = image_info[0]
            image_data = pdf.extract_image(xref)

            image_filename = (
                f"page_{page_number:03d}_image_{image_number:02d}."
                f"{image_data['ext']}"
            )

            image_path = IMAGES_DIR / image_filename
            image_path.write_bytes(image_data["image"])

            image_elements.append({
                "type": "image",
                "content": str(image_path.relative_to(DOCUMENT_DIR)),
                "metadata": {
                    "source": PDF_PATH.name,
                    "page": page_number,
                    "image_number": image_number,
                },
            })

            total_images += 1

finally:
    pdf.close()


# ==================================================
# 4. CORPUS UNIFIÉ POUR LE TRAITEMENT SUIVANT
# ==================================================

print("3. Création du manifeste unifié...")

all_elements = text_elements + table_elements + image_elements

manifest_path = PROCESSED_DIR / "extracted_data.json"

with manifest_path.open("w", encoding="utf-8") as file:
    json.dump(all_elements, file, ensure_ascii=False, indent=2)


# ==================================================
# 5. RÉSUMÉ
# ==================================================

counts = Counter(element["type"] for element in all_elements)

print("\n--- Résultat de l'extraction ---")
print(f"Pages de texte : {len(text_elements)}")
print(f"Tableaux extraits : {len(table_elements)}")
print(f"Images extraites : {total_images}")
print(f"Nombre total d'éléments : {len(all_elements)}")
print(f"Répartition : {dict(counts)}")
print(f"\nDossier du document : {DOCUMENT_DIR}")
print(f"Texte : {TEXT_DIR}")
print(f"Tableaux : {TABLES_DIR}")
print(f"Images : {IMAGES_DIR}")
print(f"Manifeste : {manifest_path}")
