"""Configuration MaintiDoc : PACIFIC par défaut, PDF importé via variables d'environnement.

Les scripts originaux continuent à fonctionner sans aucune variable d'environnement.
Chaque PDF importé dispose de ses deux collections ChromaDB indépendantes.
"""

import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
if not DATA_DIR.exists() and (BASE_DIR / "Data").exists():
    DATA_DIR = BASE_DIR / "Data"

DEFAULT_PDF_PATH = DATA_DIR / "maintenance_machine.pdf"
PDF_PATH = Path(os.environ.get("MAINTIDOC_ACTIVE_PDF", str(DEFAULT_PDF_PATH))).expanduser().resolve()
DOCUMENT_NAME = PDF_PATH.stem
DOCUMENT_DIR = DATA_DIR / "documents" / DOCUMENT_NAME
MANIFEST_PATH = DOCUMENT_DIR / "processed" / "extracted_data.json"
CHROMA_PATH = DATA_DIR / "chroma_db"

DOC_ID = os.environ.get("MAINTIDOC_DOC_ID", "").strip()
if DOC_ID and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", DOC_ID):
    raise ValueError("MAINTIDOC_DOC_ID invalide.")

# Les noms historiques restent exactement inchangés pour le manuel PACIFIC.
TEXT_COLLECTION = f"maintidoc_text_{DOC_ID}" if DOC_ID else "maintenance_pacific"
IMAGE_COLLECTION = f"maintidoc_images_{DOC_ID}" if DOC_ID else "maintenance_images"
TEXT_MODEL = "BAAI/bge-m3"
IMAGE_MODEL = "openai/clip-vit-base-patch32"
