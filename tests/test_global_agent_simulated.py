"""Test du parcours agentique global sans Ollama ni dépendances ML."""
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


class FakeTool:
    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
    def invoke(self, args):
        return self.fn(**args)


class FakeAgent:
    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}
    def invoke(self, request, config=None):
        # L'agent choisit CLIP pour une photo mais oublie BGE-M3.
        if "match_photo" in self.tools:
            self.tools["match_photo"].invoke({})
        return {"messages": [types.SimpleNamespace(content="Réponse provisoire")]}


def test():
    lc = types.ModuleType("langchain")
    agents = types.ModuleType("langchain.agents")
    agents.create_agent = lambda model, tools, system_prompt: FakeAgent(tools)
    core = types.ModuleType("langchain_core")
    core_tools = types.ModuleType("langchain_core.tools")
    core_tools.tool = FakeTool
    ollama = types.ModuleType("langchain_ollama")
    ollama.ChatOllama = lambda **kwargs: object()
    rag = types.ModuleType("rag")
    rag.generate_answer = lambda question, context: context
    image = types.ModuleType("photo_rag")
    image.analyze_photo = lambda path: "Observation visuelle incertaine"
    ret = types.ModuleType("multi_document_retrieval")
    ret.search_text = lambda docs, query, k: [
        {"document_id": "b", "document": "manuel_B.pdf", "vector_id": "b-1",
         "page": 7, "type": "text", "distance": 0.14,
         "content": "Procédure de la machine B"},
        {"document_id": "a", "document": "manuel_A.pdf", "vector_id": "a-1",
         "page": 7, "type": "table", "distance": 0.21,
         "content": "Diagnostic de la machine A"},
    ]
    ret.search_images = lambda docs, **kwargs: [
        {"document_id": "b", "document": "manuel_B.pdf", "vector_id": "b-img",
         "page": 7, "image_number": 2, "image_path": "/tmp/fake-image.jpeg", "distance": 0.0}
    ]
    ret.original_page = lambda doc, page: {
        "document_id": doc["id"], "document": doc["name"], "page": page,
        "context": "Description vérifiée dans " + doc["name"], "found": True,
    }
    ret.image_caption = lambda doc, page, number: "Légende auto non vérifiée"
    ret.format_sources = lambda hits, max_chars=10000: "\n".join(
        f"DOCUMENT : {h['document']} | PAGE : {h['page']} | {h['content']}" for h in hits)
    for mod in [lc, agents, core, core_tools, ollama, rag, image, ret]:
        sys.modules[mod.__name__] = mod
    spec = importlib.util.spec_from_file_location("multi_document_agent_fake", SRC / "multi_document_agent.py")
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    docs = [
        {"id": "a", "name": "manuel_A.pdf"},
        {"id": "b", "name": "manuel_B.pdf"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        photo = Path(tmp) / "photo.png"
        photo.write_bytes(b"fake")
        res = agent.run("Quel est ce schéma ?", docs=docs, photo=str(photo), top_k=4, image_k=2)
    trace = [e["tool"] for e in res["tool_trace"]]
    assert "match_photo" in trace
    assert "analyze_photo" in trace
    assert "search_all_manuals" in trace
    assert "read_manual_page" in trace
    assert "manuel_A.pdf" in res["answer"]
    assert "manuel_B.pdf" in res["answer"]
    assert len(res["manual_pages"]) >= 2
    print("PASS: LangGraph tools + fallback + global evidence, original-page lookup")


if __name__ == "__main__":
    test()
