"""Tests du mode global, légers et sans téléchargement des modèles ML.

Exécution : uv run python tests/test_multi_document.py
"""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pymupdf

SRC = Path(__file__).resolve().parents[1] / "src"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_search_and_sources():
    ret = load("multi_document_retrieval_test", SRC / "multi_document_retrieval.py")
    docs = [
        {"id": "pacific", "name": "maintenance_machine.pdf", "text_collection": "text_a",
         "image_collection": "img_a", "document_dir": Path("/unused/a")},
        {"id": "doc_b", "name": "guide_maintenance.pdf", "text_collection": "text_b",
         "image_collection": "img_b", "document_dir": Path("/unused/b")},
    ]
    class FakeModel:
        def get_query_embedding(self, query):
            assert query == "problème de démarrage"
            return [0.1, 0.2, 0.3]
    class FakeCollection:
        def __init__(self, name):
            self.name = name
        def count(self):
            return 2
        def query(self, query_embeddings, n_results, include):
            assert query_embeddings == [[0.1, 0.2, 0.3]]
            assert self.name in {"text_a", "text_b"}
            distance = .55 if self.name == "text_a" else .16
            return {"ids": [[self.name + "-1", self.name + "-2"]],
                    "documents": [["texte original " + self.name,
                                   "description auto " + self.name]],
                    "metadatas": [[{"page": 30, "type": "text"},
                                   {"page": 30, "type": "image_description"}]],
                    "distances": [[distance, distance + 0.1]]}
    class FakeClient:
        def get_collection(self, name):
            return FakeCollection(name)
    ret._text_model = lambda: FakeModel()
    ret._db = lambda: FakeClient()
    hits = ret.search_text(docs, "problème de démarrage", top_k=2)
    assert len(hits) == 2
    assert hits[0]["document_id"] == "doc_b"  # meilleur passage dans l'autre PDF
    assert hits[1]["document_id"] == "pacific"
    assert all(h["type"] == "text" for h in hits)
    rendered = ret.format_sources(hits)
    assert "guide_maintenance.pdf" in rendered
    assert "maintenance_machine.pdf" in rendered
    assert rendered.count("PAGE : 30") == 2  # identiques, mais bien attribuées
    # Même moteur de classement pour les images, sans mélanger les espaces.
    ranked = ret._rank_results([
        {"document_id": "a", "vector_id": "1", "distance": .2},
        {"document_id": "b", "vector_id": "1", "distance": .05},
        {"document_id": "a", "vector_id": "1", "distance": .3},
    ], 3)
    assert len(ranked) == 2 and ranked[0]["document_id"] == "b"
    print("PASS: global BGE-M3 ranking, source attribution, duplicate filtering")


def test_page_isolation():
    ret = load("multi_document_retrieval_pages", SRC / "multi_document_retrieval.py")
    with tempfile.TemporaryDirectory() as temp:
        docs = []
        for did, text in [("a", "Informations PACIFIC"), ("b", "Informations autre machine")]:
            root = Path(temp) / did
            folder = root / "processed"
            folder.mkdir(parents=True)
            (folder / "extracted_data.json").write_text(json.dumps([
                {"type": "text", "content": text, "metadata": {"page": 6}},
                {"type": "table", "content": "Pièce | Fonction", "metadata": {"page": 6}},
                {"type": "image", "description": "Description incertaine " + did,
                 "metadata": {"page": 6, "image_number": 1}},
            ]), encoding="utf-8")
            docs.append({"id": did, "name": did + ".pdf", "document_dir": root})
        pa = ret.original_page(docs[0], 6)
        pb = ret.original_page(docs[1], 6)
        assert "Informations PACIFIC" in pa["context"]
        assert "autre machine" not in pa["context"]
        assert "autre machine" in pb["context"]
        assert ret.image_caption(docs[0], 6, 1) == "Description incertaine a"
    print("PASS: same page number is isolated by document ID")


def test_worker_routing():
    with tempfile.TemporaryDirectory(prefix="maintidoc_multidoc_test_") as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        for filename in ("document_manager.py", "config.py", "multi_document_worker.py"):
            shutil.copy2(SRC / filename, root / "src" / filename)
        (root / "src" / "chromadb.py").write_text('''
class PersistentClient:
    def __init__(self, path): pass
    def get_collection(self, name):
        class Collection:
            def count(self): return 4
        return Collection()
''', encoding="utf-8")
        (root / "src" / "multi_document_agent.py").write_text('''
def run(question, *, docs, photo=None, top_k=5, image_k=3):
    return {"answer": question, "document_ids": [d["id"] for d in docs],
            "photo": photo, "top_k": top_k, "image_k": image_k,
            "tool_trace": [{"tool": "search_all_manuals"}]}
''', encoding="utf-8")
        sys.path.insert(0, str(root / "src"))  # mock Chroma également dans le processus parent
        dm = load("dm_global_test", root / "src" / "document_manager.py")
        pdf = pymupdf.open()
        pdf.new_page().insert_text((60, 70), "Maintenance")
        raw = pdf.tobytes()
        pdf.close()
        docs = []
        for label, blob in [("A.pdf", raw), ("B.pdf", raw + b"\n%Different")]:
            doc = dm.register_upload(label, blob)
            manifest = doc["document_dir"] / "processed" / "extracted_data.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("[]")
            marker = manifest.with_name("import_index_state.json")
            marker.write_text(json.dumps({"sha256": dm._file_sha256(doc["pdf_path"])}))
            docs.append(doc)
        out = dm.query_all_documents("question globale", None, top_k=4, image_k=2)
        assert set(out["document_ids"]) == {d["id"] for d in docs}
        assert out["answer"] == "question globale" and out["top_k"] == 4
        assert out["image_k"] == 2
    print("PASS: all-doc worker routing, ready-only selection, no index rebuild")


if __name__ == "__main__":
    test_search_and_sources()
    test_page_isolation()
    test_worker_routing()
