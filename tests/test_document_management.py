"""Tests unitaires locaux sans Ollama, Chroma ou modèles ML."""
import importlib.util
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pymupdf


def check():
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="maintidoc_doc_test_") as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        for name in ("config.py", "document_manager.py", "agent_worker.py"):
            shutil.copy2(source / "src" / name, root / "src" / name)

        # agent local simulé : teste réellement la transmission de l'env et des arguments.
        (root / "src" / "agent.py").write_text('''
from config import PDF_PATH, TEXT_COLLECTION, IMAGE_COLLECTION

def run(question, *, photo=None, top_k=5, image_k=3):
    print("Agent simulé prêt")
    return {"answer": question, "tool_trace": [{"tool":"search_manual"}],
            "active_pdf": str(PDF_PATH), "text_collection": TEXT_COLLECTION,
            "image_collection": IMAGE_COLLECTION, "photo": photo,
            "top_k": top_k, "image_k": image_k}
''', encoding="utf-8")
        module_spec = importlib.util.spec_from_file_location("document_manager", root / "src" / "document_manager.py")
        dm = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(dm)

        pdf = pymupdf.open()
        p = pdf.new_page()
        p.insert_text((72,72), "Document de maintenance test")
        raw = pdf.tobytes()
        pdf.close()

        d1 = dm.register_upload("Machine Test.pdf", raw)
        d1_copy = dm.register_upload("Machine Test.pdf", raw)
        assert d1["id"] == d1_copy["id"]
        assert d1["text_collection"].startswith("maintidoc_text_")
        assert d1["image_collection"].startswith("maintidoc_images_")
        assert d1["text_collection"] != "maintenance_pacific"
        assert d1["pdf_path"].exists()
        assert d1["pdf_path"].read_bytes() == raw
        assert dm.document_env(d1)["MAINTIDOC_DOC_ID"] == d1["id"]

        raw2 = raw + b"\n% different bytes"
        d2 = dm.register_upload("Machine Test.pdf", raw2)
        assert d1["id"] != d2["id"]
        assert d1["text_collection"] != d2["text_collection"]
        assert d1["image_collection"] != d2["image_collection"]

        # Émule la collection Chroma uniquement pour tester le routage JSON.
        fake_chroma = types.ModuleType("chromadb")
        class FakeCollection:
            def count(self): return 2
        class FakeClient:
            def __init__(self, path): self.path=path
            def get_collection(self, name): return FakeCollection()
        fake_chroma.PersistentClient = FakeClient
        sys.modules["chromadb"] = fake_chroma
        # L'agent_worker est un vrai subprocess avec agent.py simulé.
        for d in (d1, d2):
            manifest = d["document_dir"] / "processed" / "extracted_data.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('[]')
            state = manifest.with_name("import_index_state.json")
            state.write_text(json.dumps({"sha256": dm._file_sha256(d["pdf_path"])}))
            assert dm.is_ready(d)
            res = dm.query_agent(d, "Question test", None, top_k=4, image_k=2)
            assert res["answer"] == "Question test"
            assert res["active_pdf"] == str(d["pdf_path"])
            assert res["text_collection"] == d["text_collection"]
            assert res["image_collection"] == d["image_collection"]
            assert res["top_k"] == 4
            assert res["image_k"] == 2
        assert len(dm.list_ready_documents()) == 2
        assert not dm.DEFAULT_PDF.exists()
        print("PASS: same-PDF idempotence, SHA-256 separation, env routing, agent-worker, two-document isolation")

if __name__ == "__main__":
    check()


def check_index_cycle():
    """Valide la séquence du pipeline sans lancer les modèles lourds."""
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="maintidoc_cycle_") as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        shutil.copy2(source / "src" / "document_manager.py", root / "src" / "document_manager.py")
        spec = importlib.util.spec_from_file_location("dm_cycle", root / "src" / "document_manager.py")
        dm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dm)
        pdf = pymupdf.open()
        p = pdf.new_page()
        p.insert_text((70, 80), "test")
        raw = pdf.tobytes()
        pdf.close()
        doc = dm.register_upload("Compressor.pdf", raw)

        fake_chroma = types.ModuleType("chromadb")
        class FakeClient:
            def __init__(self, path): self.path=path
            def get_collection(self, name):
                class Collection:
                    def count(self): return 1
                return Collection()
        fake_chroma.PersistentClient = FakeClient
        sys.modules["chromadb"] = fake_chroma

        calls = []
        def fake_execute(document, name, rebuild=False):
            calls.append((name, rebuild))
            if name == "extraction.py":
                p = document["document_dir"] / "processed" / "extracted_data.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps([{"type": k} for k in ("text", "table", "image")]))
            return "simulated " + name
        dm._execute_script = fake_execute
        assert not dm.is_ready(doc)
        log = dm.index_document(doc)
        assert len(log) == 5
        assert calls == [
            ("extraction.py", True),
            ("image_captioning.py", False),
            ("chroma.py", True),
            ("image_embeddings.py", True),
            ("verify.py", False),
        ], calls
        assert dm.is_ready(doc)
        repeated = dm.index_document(doc)
        assert repeated and "aucun retraitement" in repeated[0]
        assert len(calls) == 5
        print("PASS: extraction -> captions -> BGE-M3 -> CLIP -> verify, idempotent second import")

if __name__ == "__main__":
    check_index_cycle()
