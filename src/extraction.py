"""Extraction du PDF. Utiliser --rebuild pour nettoyer les anciens artefacts du document."""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pymupdf
from llama_index.core import SimpleDirectoryReader
from config import PDF_PATH, DOCUMENT_DIR, MANIFEST_PATH


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main(rebuild=False):
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f'PDF introuvable : {PDF_PATH}')
    # Ne jamais effacer le PDF d'entrée, ni la base ChromaDB ici.
    if rebuild:
        for name in ('text', 'tables', 'images', 'processed'):
            shutil.rmtree(DOCUMENT_DIR / name, ignore_errors=True)
    original = DOCUMENT_DIR / 'original' / PDF_PATH.name
    original.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PDF_PATH, original)  # actualiser la copie, même si le PDF a changé

    elements = []
    text_dir = DOCUMENT_DIR / 'text'
    table_dir = DOCUMENT_DIR / 'tables'
    image_dir = DOCUMENT_DIR / 'images'
    for doc in SimpleDirectoryReader(input_files=[str(PDF_PATH)]).load_data():
        page = int(doc.metadata['page_label'])
        content = doc.text.strip()
        if not content:
            continue
        item = {'type': 'text', 'content': content,
                'metadata': {'source': PDF_PATH.name, 'page': page}}
        elements.append(item)
        write_json(text_dir / f'page_{page:03d}.json', item)

    with pymupdf.open(PDF_PATH) as pdf:
        for page_number, page in enumerate(pdf, 1):
            for table_number, table in enumerate(page.find_tables().tables, 1):
                rows = [[str(c).strip() if c is not None else '' for c in row]
                        for row in table.extract()]
                content = '\n'.join(' | '.join(row) for row in rows)
                if not content.strip():
                    continue
                item = {'type': 'table', 'content': content, 'rows': rows,
                        'metadata': {'source': PDF_PATH.name, 'page': page_number,
                                     'table_number': table_number}}
                elements.append(item)
                write_json(table_dir / f'page_{page_number:03d}_table_{table_number:02d}.json', item)
            for image_number, info in enumerate(page.get_images(full=True), 1):
                extracted = pdf.extract_image(info[0])
                if not extracted or not extracted.get('image'):
                    continue
                filename = f'page_{page_number:03d}_image_{image_number:02d}.{extracted["ext"]}'
                image_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / filename).write_bytes(extracted['image'])
                elements.append({'type': 'image', 'content': f'images/{filename}',
                                 'metadata': {'source': PDF_PATH.name, 'page': page_number,
                                              'image_number': image_number}})

    write_json(MANIFEST_PATH, elements)
    counts = Counter(e['type'] for e in elements)
    print(f'Manifeste : {MANIFEST_PATH}\nÉléments : {len(elements)} ; répartition : {dict(counts)}')
    print('Attention : les tableaux sont aussi susceptibles de figurer dans le texte des pages.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='Nettoyer les sorties de ce PDF avant extraction')
    main(parser.parse_args().rebuild)
