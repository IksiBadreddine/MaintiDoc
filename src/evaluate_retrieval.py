
"""Évaluation du retrieval textuel de MaintiDoc.

Métriques :
- Hit@k : au moins une page attendue retrouvée
- Recall@k : proportion des pages attendues retrouvées
- MRR@k : inverse du rang du premier résultat pertinent
"""

import json
from pathlib import Path

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_PATH = BASE_DIR / "data" / "evaluation" / "questions.json"
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "maintenance_pacific"
MODEL_NAME = "BAAI/bge-m3"

TOP_K_VALUES = [3, 5, 10]


# ============================================================
# 2. CHARGEMENT DES QUESTIONS
# ============================================================

def load_questions():
    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("questions.json doit contenir une liste.")

    if not questions:
        raise ValueError("Le fichier questions.json est vide.")

    for question in questions:
        if not isinstance(question.get("question"), str):
            raise ValueError("Chaque entrée doit avoir une question.")

        pages = question.get("expected_pages")

        if not isinstance(pages, list):
            raise ValueError("expected_pages doit être une liste.")

        if not all(isinstance(p, int) and p > 0 for p in pages):
            raise ValueError("Les pages attendues doivent être des entiers positifs.")

    return questions


# ============================================================
# 3. CHARGEMENT DU RETRIEVER
# ============================================================

def load_retriever():
    print("Chargement de BGE-M3...")

    embed_model = HuggingFaceEmbedding(
        model_name=MODEL_NAME
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    retriever = index.as_retriever(
        similarity_top_k=max(TOP_K_VALUES)
    )

    print(f"Collection : {COLLECTION_NAME}")
    print(f"Vecteurs disponibles : {collection.count()}")

    return retriever


# ============================================================
# 4. CALCUL DES MÉTRIQUES
# ============================================================

def calculate_metrics(results, expected_pages, k):
    """Calcule les métriques au niveau des pages."""

    top_results = results[:k]

    retrieved_pages = [
        int(result.node.metadata["page"])
        for result in top_results
    ]

    expected = set(expected_pages)

    if not expected:
        return None

    # Pages attendues retrouvées dans les k premiers chunks
    found = expected.intersection(retrieved_pages)

    # Hit@k : 1 si au moins une page correcte est retrouvée
    hit = 1.0 if found else 0.0

    # Recall@k : proportion des pages attendues retrouvées
    recall = len(found) / len(expected)

    # MRR@k : rang du premier chunk provenant d'une page attendue
    reciprocal_rank = 0.0

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "hit": hit,
        "recall": recall,
        "mrr": reciprocal_rank,
    }


# ============================================================
# 5. ÉVALUATION
# ============================================================

def evaluate():
    questions = load_questions()
    retriever = load_retriever()

    scores = {
        k: {"hit": [], "recall": [], "mrr": []}
        for k in TOP_K_VALUES
    }

    evaluated = 0

    for item in questions:
        question = item["question"]
        expected_pages = item["expected_pages"]

        print("\n" + "=" * 65)
        print(f"Question {item.get('id', '?')} : {question}")
        print(f"Pages attendues : {expected_pages}")

        # Une seule recherche avec top_k = 10
        results = retriever.retrieve(question)

        print("\nRésultats du retriever :")

        for rank, result in enumerate(results, start=1):
            page = int(result.node.metadata["page"])
            element_type = result.node.metadata.get("type", "?")
            score = result.score

            score_text = (
                f"{score:.4f}"
                if score is not None
                else "N/A"
            )

            correct = page in expected_pages
            marker = "✓" if correct else " "

            print(
                f"{rank:2d}. Page {page:2d} "
                f"| Type : {element_type:5s} "
                f"| Score : {score_text} "
                f"| {marker}"
            )

        if not expected_pages:
            print(
                "\nQuestion sans page attendue : "
                "exclue des métriques positives."
            )
            continue

        evaluated += 1

        print("\nMétriques :")

        for k in TOP_K_VALUES:
            metrics = calculate_metrics(
                results, expected_pages, k
            )

            for metric_name, value in metrics.items():
                scores[k][metric_name].append(value)

            print(
                f"@{k:2d} | "
                f"Hit = {metrics['hit']:.2f} | "
                f"Recall = {metrics['recall']:.2f} | "
                f"MRR = {metrics['mrr']:.2f}"
            )

    # ========================================================
    # 6. RÉSULTATS GLOBAUX
    # ========================================================

    print("\n" + "=" * 65)
    print("RÉSULTATS GLOBAUX")
    print("=" * 65)

    if evaluated == 0:
        print("Aucune question positive à évaluer.")
        return

    print(f"Questions évaluées : {evaluated}")

    for k in TOP_K_VALUES:
        averages = {
            name: sum(values) / len(values)
            for name, values in scores[k].items()
        }

        print(
            f"\nTop-{k}"
            f"\n  Hit@{k}    : {averages['hit']:.4f}"
            f"\n  Recall@{k} : {averages['recall']:.4f}"
            f"\n  MRR@{k}    : {averages['mrr']:.4f}"
        )


if __name__ == "__main__":
    evaluate()
