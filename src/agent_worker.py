"""Exécute l'agent avec le document actif fixé par document_manager.py.

Ce script démarre dans un processus isolé, pour éviter de conserver en mémoire
une configuration ou un retriever d'un autre document Streamlit.
"""
import json
import sys

from agent import run


def main():
    params = json.load(sys.stdin)
    if not isinstance(params, dict):
        raise ValueError("Requête invalide.")
    result = run(
        params["question"],
        photo=params.get("photo"),
        top_k=int(params.get("top_k", 5)),
        image_k=int(params.get("image_k", 3)),
    )
    # Dernière ligne structurée, après les logs d'Ollama, CLIP et LangGraph.
    print("__MAINTIDOC_JSON_RESULT__=" + json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
