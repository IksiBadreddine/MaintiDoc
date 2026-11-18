"""Gestion des PDF importés depuis Streamlit, sans modifier le manuel PACIFIC.

Chaque PDF est identifié par son nom normalisé et son SHA-256, avec :
  data/imported/<id>.pdf
  data/documents/<id>/... (extraction, manifeste, captions)
  ChromaDB maintidoc_text_<id> et maintidoc_images_<id>

Les scripts historiques sont lancés dans des processus avec des variables
d'environnement propres au PDF actif. Pas de mutation globale de config.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
if not DATA.exists() and (ROOT / "Data").exists():
    DATA = ROOT / "Data"
DEFAULT_PDF = DATA / "maintenance_machine.pdf"
IMPORTS = DATA / "imported"
DOCUMENTS = DATA / "documents"
CHROMA = DATA / "chroma_db"
MAX_PDF_SIZE = 25 * 1024 * 1024
DOC_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}\Z")
RESULT_MARKER = "__MAINTIDOC_JSON_RESULT__="


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(filename: str) -> str:
    stem = Path(filename.replace("\\", "/")).name.rsplit(".", 1)[0]
    clean = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", clean.lower()).strip("_")[:27] or "manuel"


def _record(document_id: str) -> dict:
    if document_id == "pacific":
        if not DEFAULT_PDF.is_file():
            raise FileNotFoundError("Le manuel PACIFIC est introuvable.")
        return {
            "id": "pacific",
            "name": DEFAULT_PDF.name,
            "pdf_path": DEFAULT_PDF,
            "document_dir": DOCUMENTS / DEFAULT_PDF.stem,
            "text_collection": "maintenance_pacific",
            "image_collection": "maintenance_images",
            "env_id": "",
        }
    if not DOC_ID_RE.fullmatch(document_id):
        raise ValueError("Identifiant de document invalide.")
    pdf = IMPORTS / f"{document_id}.pdf"
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF importé introuvable : {pdf}")
    metadata_path = IMPORTS / f"{document_id}.json"
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    return {
        "id": document_id,
        "name": str(meta.get("original_name") or pdf.name),
        "pdf_path": pdf,
        "document_dir": DOCUMENTS / document_id,
        "text_collection": f"maintidoc_text_{document_id}",
        "image_collection": f"maintidoc_images_{document_id}",
        "env_id": document_id,
    }


def document_env(doc: dict) -> dict[str, str]:
    env = dict(os.environ)
    env["MAINTIDOC_ACTIVE_PDF"] = str(doc["pdf_path"].resolve())
    if doc["env_id"]:
        env["MAINTIDOC_DOC_ID"] = doc["env_id"]
    else:
        env.pop("MAINTIDOC_DOC_ID", None)
    return env


def register_upload(filename: str, raw: bytes) -> dict:
    """Valide un upload PDF et l'enregistre sans lancer son indexation."""
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Sélectionne un fichier .pdf.")
    if not raw or len(raw) > MAX_PDF_SIZE:
        raise ValueError("Le PDF doit être non vide et ne pas dépasser 25 Mo.")

    # Vérification du document avant toute écriture sur disque.
    import pymupdf
    try:
        with pymupdf.open(stream=raw, filetype="pdf") as document:
            if document.is_encrypted:
                raise ValueError("Les PDF protégés par mot de passe ne sont pas pris en charge.")
            if len(document) == 0 or len(document) > 1000:
                raise ValueError("Le PDF doit contenir entre 1 et 1000 pages.")
    except (RuntimeError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Fichier PDF invalide ou illisible.") from exc

    digest = _sha256(raw)
    if DEFAULT_PDF.is_file() and _file_sha256(DEFAULT_PDF) == digest:
        return _record("pacific")

    document_id = f"{_slug(filename)}_{digest[:16]}"
    if not DOC_ID_RE.fullmatch(document_id):
        raise ValueError("Le nom normalisé du document est invalide.")
    IMPORTS.mkdir(parents=True, exist_ok=True)
    path = IMPORTS / f"{document_id}.pdf"
    if path.is_file():
        if _file_sha256(path) != digest:
            raise RuntimeError("Conflit de fichiers : le document enregistré a changé.")
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)

    info = IMPORTS / f"{document_id}.json"
    if not info.exists():
        tmp = info.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"original_name": Path(filename.replace('\\', '/')).name, "sha256": digest}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(info)
    return _record(document_id)


def is_ready(doc: dict) -> bool:
    manifest = doc["document_dir"] / "processed" / "extracted_data.json"
    if not manifest.is_file():
        return False
    if doc["id"] != "pacific":
        marker = doc["document_dir"] / "processed" / "import_index_state.json"
        try:
            state = json.loads(marker.read_text(encoding="utf-8"))
            if state.get("sha256") != _file_sha256(doc["pdf_path"]):
                return False
        except (OSError, ValueError):
            return False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA))
        return all(
            client.get_collection(name=name).count() > 0
            for name in (doc["text_collection"], doc["image_collection"])
        )
    except Exception:
        return False


def list_ready_documents() -> list[dict]:
    docs = []
    if DEFAULT_PDF.is_file():
        candidate = _record("pacific")
        if is_ready(candidate):
            docs.append(candidate)
    if IMPORTS.exists():
        for path in sorted(IMPORTS.glob("*.pdf")):
            if not DOC_ID_RE.fullmatch(path.stem):
                continue
            doc = _record(path.stem)
            if is_ready(doc):
                docs.append(doc)
    return docs


def _execute_script(doc: dict, script: str, rebuild: bool = False) -> str:
    command = [sys.executable, "-u", str(SRC / script)]
    if rebuild:
        command.append("--rebuild")
    result = subprocess.run(
        command, cwd=ROOT, env=document_env(doc), capture_output=True,
        text=True, timeout=3600, check=False,
    )
    if result.returncode:
        detail = "\n".join(x for x in (result.stdout[-5000:], result.stderr[-5000:]) if x)
        raise RuntimeError(f"Échec de {script} :\n{detail}")
    return result.stdout


def index_document(doc: dict) -> list[str]:
    """Exécute extraction -> captions -> BGE-M3 -> CLIP -> validation une fois."""
    if is_ready(doc):
        return ["Le PDF possède déjà un index valide : aucun retraitement."]
    if doc["id"] == "pacific":
        raise RuntimeError("Index PACIFIC absent : exécute d'abord 'uv run python main.py'.")

    logs = []
    logs.append(_execute_script(doc, "extraction.py", rebuild=True))
    manifest = doc["document_dir"] / "processed" / "extracted_data.json"
    elements = json.loads(manifest.read_text(encoding="utf-8"))
    counts = Counter(e.get("type") for e in elements)
    missing = [kind for kind in ("text", "image", "table") if counts[kind] == 0]
    if missing:
        raise ValueError(
            "Ce PDF ne fournit pas les trois modalités demandées (texte, image, tableau). "
            "Types manquants dans l'extraction : " + ", ".join(missing) + ". "
            "Les tableaux ou images doivent être extractibles avec le pipeline actuel."
        )

    logs.append(_execute_script(doc, "image_captioning.py"))
    logs.append(_execute_script(doc, "chroma.py", rebuild=True))
    logs.append(_execute_script(doc, "image_embeddings.py", rebuild=True))
    logs.append(_execute_script(doc, "verify.py"))

    state = doc["document_dir"] / "processed" / "import_index_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    tmp = state.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({
            "sha256": _file_sha256(doc["pdf_path"]),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "counts": dict(counts),
            "collections": [doc["text_collection"], doc["image_collection"]],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(state)
    if not is_ready(doc):
        raise RuntimeError("L'index a été créé mais n'est pas reconnu comme prêt.")
    return logs


def query_agent(doc: dict, question: str, photo: str | None, top_k: int, image_k: int) -> dict:
    """Interroge l'agent dans un processus isolé configuré pour CE document."""
    if not is_ready(doc):
        raise RuntimeError("Le document sélectionné n'est pas encore indexé.")
    request = {"question": question, "photo": photo, "top_k": top_k, "image_k": image_k}
    result = subprocess.run(
        [sys.executable, "-u", str(SRC / "agent_worker.py")],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, cwd=ROOT, env=document_env(doc), timeout=900,
        check=False,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    raise RuntimeError("L'agent n'a pas renvoyé de résultat JSON valide.\n" + result.stdout[-1500:])


def query_all_documents(question: str, photo: str | None,
                        top_k: int, image_k: int) -> dict:
    """Recherche globale sur TOUS les PDF prêts, dans un unique agent worker.

    Les embeddings et le captioning des manuels ne sont pas recalculés.
    """
    docs = list_ready_documents()
    if not docs:
        raise RuntimeError("Aucun document indexé prêt pour la recherche globale.")
    if top_k < 1 or image_k < 1:
        raise ValueError("top_k et image_k doivent être >= 1")
    request = {
        "question": question,
        "photo": photo,
        "top_k": top_k,
        "image_k": image_k,
        "document_ids": [doc["id"] for doc in docs],
    }
    # Le worker inspecte les identifiants contre la liste locale autorisée.
    global_env = dict(os.environ)
    global_env.pop("MAINTIDOC_DOC_ID", None)
    global_env.pop("MAINTIDOC_ACTIVE_PDF", None)
    result = subprocess.run(
        [sys.executable, "-u", str(SRC / "multi_document_worker.py")],
        input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, cwd=ROOT,
        env=global_env,
        timeout=900, check=False,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-4000:])
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])
    raise RuntimeError("Aucun résultat JSON envoyé par l'agent global :\n" + result.stdout[-1500:])
