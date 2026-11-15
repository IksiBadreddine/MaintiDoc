
"""Pipeline MaintiDoc avec traitement unique du PDF.

Commandes :
    uv run python main.py --adopt-existing
    uv run python main.py
    uv run python main.py --rebuild
    uv run python main.py --with-query

Principe :
    - PDF nouveau : traitement complet
    - PDF inchangé : réutilisation des résultats
    - PDF modifié : reconstruction
    - --rebuild : reconstruction forcée
    - --adopt-existing : adoption des données déjà indexées
"""

import argparse
import hashlib
import json
import subprocess
import sys

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

PIPELINE_VERSION = 1


# ============================================================
# LOCALISATION DES DONNÉES
# ============================================================

def locate_data_dir():
    """Localise le dossier data/ ou Data/."""

    candidates = [
        path
        for path in (
            ROOT / "data",
            ROOT / "Data",
        )
        if path.is_dir()
    ]

    if not candidates:
        raise FileNotFoundError(
            "Aucun dossier data/ ou Data/ trouvé."
        )

    if (
        len(candidates) == 2
        and not candidates[0].samefile(candidates[1])
    ):
        raise RuntimeError(
            "Deux dossiers de données distincts existent."
        )

    return candidates[0]


# ============================================================
# EXÉCUTION DES SCRIPTS
# ============================================================

def run_script(filename, rebuild=False):
    """Exécute un script du pipeline."""

    script = SRC / filename

    if not script.is_file():
        raise FileNotFoundError(
            f"Script manquant : {script}"
        )

    command = [
        sys.executable,
        "-u",
        str(script),
    ]

    if rebuild:
        command.append("--rebuild")

    print(
        f"\n{'=' * 65}\n"
        f"Exécution : {filename}"
        f"{' --rebuild' if rebuild else ''}\n"
        f"{'=' * 65}",
        flush=True,
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


# ============================================================
# EMPREINTE SHA-256
# ============================================================

def pdf_fingerprint(pdf):
    """Calcule l'empreinte SHA-256 du PDF."""

    digest = hashlib.sha256()

    with pdf.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# ============================================================
# GESTION DE L'ÉTAT PERSISTANT
# ============================================================

def state_path(data_dir, pdf):
    """Retourne le chemin du fichier d'état."""

    return (
        data_dir
        / "documents"
        / pdf.stem
        / "processed"
        / "pipeline_state.json"
    )


def read_state(path):
    """Charge l'état précédent du pipeline."""

    if not path.is_file():
        return None

    try:
        state = json.loads(
            path.read_text(encoding="utf-8")
        )

        if not isinstance(state, dict):
            return None

        return state

    except (OSError, ValueError):
        return None


def save_state(path, pdf):
    """Enregistre l'état après validation complète."""

    state = {
        "pdf_name": pdf.name,
        "pdf_sha256": pdf_fingerprint(pdf),
        "status": "completed",
        "pipeline_version": PIPELINE_VERSION,
        "completed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


# ============================================================
# MANIFESTE
# ============================================================

def load_manifest(data_dir, pdf_name):
    """Charge le manifeste d'extraction."""

    doc_dir = (
        data_dir
        / "documents"
        / Path(pdf_name).stem
    )

    manifest = (
        doc_dir
        / "processed"
        / "extracted_data.json"
    )

    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifeste absent : {manifest}"
        )

    elements = json.loads(
        manifest.read_text(encoding="utf-8")
    )

    if (
        not isinstance(elements, list)
        or not elements
    ):
        raise ValueError(
            "Manifeste vide ou invalide."
        )

    return doc_dir, elements


# ============================================================
# VALIDATION DE L'EXTRACTION
# ============================================================

def validate_extraction(data_dir, pdf_name):
    """Vérifie les éléments et les fichiers extraits."""

    doc_dir, elements = load_manifest(
        data_dir,
        pdf_name,
    )

    counts = Counter()
    seen = set()

    for index, element in enumerate(
        elements,
        start=1,
    ):

        kind = element.get("type")
        content = element.get("content")
        metadata = element.get("metadata")

        if kind not in {
            "text",
            "table",
            "image",
        }:
            raise ValueError(
                f"Élément {index} : type invalide."
            )

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError(
                f"Élément {index} : contenu invalide."
            )

        if (
            not isinstance(metadata, dict)
            or metadata.get("source") != pdf_name
        ):
            raise ValueError(
                f"Élément {index} : métadonnées invalides."
            )

        page = metadata.get("page")

        if (
            type(page) is not int
            or page < 1
        ):
            raise ValueError(
                f"Élément {index} : page invalide."
            )

        # ---------------- TEXTES ----------------

        if kind == "text":

            file = (
                doc_dir
                / "text"
                / f"page_{page:03d}.json"
            )

            key = (kind, page)

        # ---------------- TABLEAUX ----------------

        elif kind == "table":

            number = metadata.get(
                "table_number"
            )

            if (
                type(number) is not int
                or number < 1
                or not isinstance(
                    element.get("rows"),
                    list,
                )
            ):
                raise ValueError(
                    f"Élément {index} : tableau invalide."
                )

            file = (
                doc_dir
                / "tables"
                / (
                    f"page_{page:03d}"
                    f"_table_{number:02d}.json"
                )
            )

            key = (
                kind,
                page,
                number,
            )

        # ---------------- IMAGES ----------------

        else:

            number = metadata.get(
                "image_number"
            )

            if (
                type(number) is not int
                or number < 1
            ):
                raise ValueError(
                    f"Élément {index} : image invalide."
                )

            file = (
                doc_dir / content
            ).resolve()

            image_dir = (
                doc_dir / "images"
            ).resolve()

            if not file.is_relative_to(
                image_dir
            ):
                raise ValueError(
                    f"Élément {index} : chemin interdit."
                )

            if not isinstance(
                element.get("context"),
                str,
            ):
                raise ValueError(
                    f"Élément {index} : contexte absent."
                )

            key = (
                kind,
                page,
                number,
            )

        # ---------------- CONTRÔLES ----------------

        if key in seen:
            raise ValueError(
                f"Élément dupliqué : {key}"
            )

        seen.add(key)

        if not file.is_file():
            raise FileNotFoundError(
                f"Fichier absent : {file}"
            )

        counts[kind] += 1

    for kind in (
        "text",
        "table",
        "image",
    ):
        if counts[kind] == 0:
            raise ValueError(
                f"Aucun élément de type {kind}."
            )

    print(
        "\nVALIDATION EXTRACTION : "
        f"{dict(counts)}",
        flush=True,
    )

    return counts


# ============================================================
# VALIDATION DES DESCRIPTIONS
# ============================================================

def validate_captions(data_dir, pdf_name):
    """Vérifie les descriptions de toutes les images."""

    _, elements = load_manifest(
        data_dir,
        pdf_name,
    )

    images = [
        element
        for element in elements
        if element["type"] == "image"
    ]

    completed = [
        element
        for element in images
        if (
            element.get("caption_status")
            == "completed"
            and isinstance(
                element.get("description"),
                str,
            )
            and element["description"].strip()
        )
    ]

    print(
        "\nVALIDATION DES DESCRIPTIONS",
        flush=True,
    )

    print(
        f"Images : {len(images)}",
        flush=True,
    )

    print(
        f"Descriptions : {len(completed)}",
        flush=True,
    )

    if len(completed) != len(images):
        raise RuntimeError(
            "Certaines images n'ont pas "
            "de description valide."
        )


# ============================================================
# VÉRIFICATION DES DONNÉES EXISTANTES
# ============================================================

def verify_existing(data_dir, pdf):
    """Vérifie les artefacts et les index persistants."""

    validate_extraction(
        data_dir,
        pdf.name,
    )

    validate_captions(
        data_dir,
        pdf.name,
    )

    run_script("verify.py")


# ============================================================
# TRAITEMENT COMPLET
# ============================================================

def process_document(data_dir, pdf):
    """Exécute extraction, captioning et indexation."""

    # 1. EXTRACTION

    print(
        "\n[1/5] EXTRACTION DU PDF",
        flush=True,
    )

    run_script(
        "extraction.py",
        rebuild=True,
    )

    validate_extraction(
        data_dir,
        pdf.name,
    )

    # 2. DESCRIPTION DES IMAGES

    print(
        "\n[2/5] DESCRIPTION DES IMAGES",
        flush=True,
    )

    run_script(
        "image_captioning.py"
    )

    validate_captions(
        data_dir,
        pdf.name,
    )

    # 3. INDEXATION BGE-M3

    print(
        "\n[3/5] INDEXATION TEXTUELLE BGE-M3",
        flush=True,
    )

    run_script(
        "chroma.py",
        rebuild=True,
    )

    # 4. INDEXATION CLIP

    print(
        "\n[4/5] INDEXATION VISUELLE CLIP",
        flush=True,
    )

    run_script(
        "image_embeddings.py",
        rebuild=True,
    )

    # 5. VÉRIFICATION

    print(
        "\n[5/5] VÉRIFICATION GLOBALE",
        flush=True,
    )

    run_script(
        "verify.py"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline MaintiDoc avec "
            "traitement unique par PDF"
        )
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruire complètement le pipeline.",
    )

    parser.add_argument(
        "--with-query",
        action="store_true",
        help="Exécuter la recherche de démonstration.",
    )

    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help="Adopter les données déjà générées.",
    )

    args = parser.parse_args()

    if args.rebuild and args.adopt_existing:
        parser.error(
            "--rebuild et --adopt-existing "
            "ne peuvent pas être utilisés ensemble."
        )

    # ---------------- LOCALISATION ----------------

    data_dir = locate_data_dir()

    pdf = (
        data_dir
        / "maintenance_machine.pdf"
    )

    if not pdf.is_file():
        raise FileNotFoundError(
            f"PDF introuvable : {pdf}"
        )

    path = state_path(
        data_dir,
        pdf,
    )

    state = read_state(path)

    fingerprint = pdf_fingerprint(pdf)

    print(
        f"\nPDF source : {pdf}",
        flush=True,
    )

    # ---------------- RECONSTRUCTION FORCÉE ----------------

    if args.rebuild:

        print(
            "\nReconstruction complète demandée.",
            flush=True,
        )

        path.unlink(
            missing_ok=True,
        )

        process_document(
            data_dir,
            pdf,
        )

        save_state(
            path,
            pdf,
        )

    # ---------------- DOCUMENT DÉJÀ TRAITÉ ----------------

    elif (
        state is not None
        and state.get("status") == "completed"
        and state.get("pdf_name") == pdf.name
        and state.get("pdf_sha256") == fingerprint
        and state.get("pipeline_version")
        == PIPELINE_VERSION
    ):

        print(
            "\nDocument déjà traité et inchangé.",
            flush=True,
        )

        print(
            "Réutilisation des données persistantes.",
            flush=True,
        )

        verify_existing(
            data_dir,
            pdf,
        )

        print(
            "\nAucune extraction nécessaire.",
            flush=True,
        )

        print(
            "Aucun captioning nécessaire.",
            flush=True,
        )

        print(
            "Aucune indexation nécessaire.",
            flush=True,
        )

    # ---------------- ADOPTION DES DONNÉES ----------------

    elif args.adopt_existing:

        if state is not None:
            if (
                state.get("pdf_sha256")
                != fingerprint
            ):
                raise RuntimeError(
                    "Le PDF a changé depuis "
                    "le dernier traitement. "
                    "Utiliser --rebuild."
                )

        print(
            "\nAdoption des résultats existants...",
            flush=True,
        )

        verify_existing(
            data_dir,
            pdf,
        )

        save_state(
            path,
            pdf,
        )

        print(
            "\nDonnées existantes adoptées.",
            flush=True,
        )

    # ---------------- PREMIER TRAITEMENT ----------------

    else:

        print(
            "\nAucun traitement valide enregistré.",
            flush=True,
        )

        print(
            "Démarrage du pipeline complet...",
            flush=True,
        )

        path.unlink(
            missing_ok=True,
        )

        process_document(
            data_dir,
            pdf,
        )

        save_state(
            path,
            pdf,
        )

    # ---------------- RECHERCHE OPTIONNELLE ----------------

    if args.with_query:

        print(
            "\nLancement de la recherche...",
            flush=True,
        )

        run_script(
            "query.py"
        )

    print(
        "\n" + "=" * 65,
        flush=True,
    )

    print(
        "PIPELINE MAINTIDOC TERMINÉ",
        flush=True,
    )

    print(
        "=" * 65,
        flush=True,
    )


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except (
        Exception,
        subprocess.CalledProcessError,
    ) as error:

        print(
            f"\nPIPELINE ARRÊTÉ : {error}",
            file=sys.stderr,
            flush=True,
        )

        sys.exit(1)
