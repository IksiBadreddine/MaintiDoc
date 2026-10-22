from pathlib import Path
from llama_index.core import SimpleDirectoryReader


PDF_PATH = Path("data/maintenance_machine.pdf")


documents = SimpleDirectoryReader(
    input_files=[str(PDF_PATH)]
).load_data()


print(f"Nombre de documents extraits : {len(documents)}")

for i, document in enumerate(documents[:3]):

    print(f"\n{'=' * 60}")
    print(f"DOCUMENT {i + 1}")
    print(f"{'=' * 60}")

    print("\n--- Métadonnées ---")
    print(document.metadata)

    print("\n--- Texte ---")
    print(document.text[:500])