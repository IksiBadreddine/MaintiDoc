"""Point d'entrée unique du pipeline MaintiDoc (extraction -> indexation -> vérification).

Première exécution : uv run python main.py
Reconstruction :    uv run python main.py --rebuild
Recherche facultative après validation : uv run python main.py --with-query
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def locate_data_dir() -> Path:
    """Trouve le dossier de données, qu'il soit nommé data ou Data."""
    candidates = [p for p in (ROOT / "data", ROOT / "Data") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("Aucun dossier data/ ou Data/ à la racine du projet.")
    if len(candidates) == 2 and not candidates[0].samefile(candidates[1]):
        raise RuntimeError("Deux dossiers distincts data/ et Data/ existent : harmoniser les chemins.")
    return candidates[0]


def run_script(filename: str, *, rebuild: bool = False) -> None:
    script = SRC / filename
    if not script.is_file():
        raise FileNotFoundError(f"Script manquant : {script}")
    command = [sys.executable, "-u", str(script)]
    if rebuild:
        command.append("--rebuild")
    print(f"\n{'=' * 65}\nExécution : {filename}{' --rebuild' if rebuild else ''}\n{'=' * 65}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def validate_extraction(data_dir: Path, pdf_name: str) -> Counter:
    """Contrôle la cohérence du manifeste ET des fichiers générés avant indexation."""
    doc_dir = data_dir / "documents" / Path(pdf_name).stem
    manifest = doc_dir / "processed" / "extracted_data.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifeste absent : {manifest}")
    elements = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(elements, list) or not elements:
        raise ValueError("Le manifeste doit contenir une liste non vide.")

    counts = Counter()
    seen = set()
    for i, element in enumerate(elements, 1):
        kind = element.get("type")
        content = element.get("content")
        metadata = element.get("metadata")
        if kind not in {"text", "table", "image"}:
            raise ValueError(f"Élément {i} : type incorrect ({kind!r})")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Élément {i} : contenu vide ou invalide")
        if not isinstance(metadata, dict) or metadata.get("source") != pdf_name:
            raise ValueError(f"Élément {i} : métadonnées/source incorrectes")
        page = metadata.get("page")
        if type(page) is not int or page < 1:
            raise ValueError(f"Élément {i} : numéro de page incorrect")

        if kind == "text":
            file = doc_dir / "text" / f"page_{page:03d}.json"
            key = (kind, page)
        elif kind == "table":
            number = metadata.get("table_number")
            if type(number) is not int or number < 1 or not isinstance(element.get("rows"), list):
                raise ValueError(f"Élément {i} : numéro ou structure du tableau invalide")
            file = doc_dir / "tables" / f"page_{page:03d}_table_{number:02d}.json"
            key = (kind, page, number)
        else:
            number = metadata.get("image_number")
            if type(number) is not int or number < 1:
                raise ValueError(f"Élément {i} : numéro d'image invalide")
            file = (doc_dir / content).resolve()
            if not file.is_relative_to((doc_dir / "images").resolve()):
                raise ValueError(f"Élément {i} : chemin d'image hors du dossier images")
            key = (kind, page, number)

        if key in seen:
            raise ValueError(f"Élément dupliqué : {key}")
        seen.add(key)
        if not file.is_file():
            raise FileNotFoundError(f"Élément {i} : fichier absent : {file}")
        counts[kind] += 1

    for kind in ("text", "table", "image"):
        if counts[kind] == 0:
            raise ValueError(f"Aucun élément de type {kind} : vérifier le PDF et l'extraction.")

    print(f"\nValidation de l'extraction : {dict(counts)} ; total : {sum(counts.values())}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline complet d'extraction et d'indexation MaintiDoc")
    parser.add_argument("--rebuild", action="store_true", help="Remplacer les artefacts et les deux collections existantes")
    parser.add_argument("--with-query", action="store_true", help="Lancer la recherche de démonstration après validation")
    args = parser.parse_args()

    data_dir = locate_data_dir()
    pdf = data_dir / "maintenance_machine.pdf"
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF source introuvable : {pdf}")
    print(f"PDF source : {pdf}", flush=True)

    # Un seul point d'entrée ; chaque script reste autonome et réutilisable.
    run_script("extraction.py", rebuild=args.rebuild)
    validate_extraction(data_dir, pdf.name)
    run_script("chroma.py", rebuild=args.rebuild)
    run_script("image_embeddings.py", rebuild=args.rebuild)
    run_script("verify.py")
    if args.with_query:
        run_script("query.py")
    print("\nPIPELINE TERMINÉ : extraction, deux index et vérifications réussis.", flush=True)
    print("La qualité sémantique des tableaux et images reste à inspecter sur le PDF.")


if __name__ == "__main__":
    try:
        main()
    except (Exception, subprocess.CalledProcessError) as exc:
        print(f"\nPIPELINE ARRÊTÉ : {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
