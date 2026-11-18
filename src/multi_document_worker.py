"""Worker isolé : effectue une interrogation globale sans modifier les index."""
import json
import sys

from document_manager import list_ready_documents, RESULT_MARKER
from multi_document_agent import run


def main():
    request = json.load(sys.stdin)
    # Ne jamais faire confiance à un chemin ni à une collection du client.
    ids = set(request["document_ids"])
    all_docs = list_ready_documents()
    docs = [d for d in all_docs if d["id"] in ids]
    if not ids or {d["id"] for d in docs} != ids:
        raise ValueError("Sélection de documents invalide ou incomplète")
    result = run(
        request["question"], docs=docs,
        photo=request.get("photo"),
        top_k=int(request.get("top_k", 5)),
        image_k=int(request.get("image_k", 3)),
    )
    print(RESULT_MARKER + json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
