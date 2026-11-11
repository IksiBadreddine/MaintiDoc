from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
# On macOS, 'Data' et 'data' peuvent désigner le même dossier.
if not DATA_DIR.exists() and (BASE_DIR / 'Data').exists():
    DATA_DIR = BASE_DIR / 'Data'

PDF_PATH = DATA_DIR / 'maintenance_machine.pdf'
DOCUMENT_NAME = PDF_PATH.stem
DOCUMENT_DIR = DATA_DIR / 'documents' / DOCUMENT_NAME
MANIFEST_PATH = DOCUMENT_DIR / 'processed' / 'extracted_data.json'
CHROMA_PATH = DATA_DIR / 'chroma_db'
TEXT_COLLECTION = 'maintenance_pacific'
IMAGE_COLLECTION = 'maintenance_images'
TEXT_MODEL = 'BAAI/bge-m3'
IMAGE_MODEL = 'openai/clip-vit-base-patch32'
