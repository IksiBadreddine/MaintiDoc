# MaintiDoc

MaintiDoc est un prototype Python d'extraction et de recherche dans de la
documentation de maintenance. Le projet prépare les contenus d'un manuel PDF,
transforme les textes et tableaux en vecteurs de recherche sémantique et crée
séparément des vecteurs pour les images.

> **État actuel :** les scripts de traitement sont présents, mais ils ne sont
> pas encore réunis dans une application ou une commande unique. `main.py` est
> encore un point d'entrée de démonstration qui affiche seulement
> `Hello from maintidoc!`. Pour le moment, les scripts de `src/` se lancent
> individuellement, dans l'ordre indiqué ci-dessous.

## Sommaire

- [Objectif et fonctionnement global](#objectif-et-fonctionnement-global)
- [Contenu du dépôt](#contenu-du-dépôt)
- [Technologies](#technologies)
- [Installation](#installation)
- [Pipeline de traitement](#pipeline-de-traitement)
- [Collections ChromaDB](#collections-chromadb)
- [Données et formats](#données-et-formats)
- [Limites et points d'attention](#limites-et-points-dattention)

## Objectif et fonctionnement global

Le projet met en place les étapes suivantes pour le manuel de maintenance de la
machine :

1. Lire un PDF et extraire le texte page par page, les tableaux détectés et les
   images intégrées.
2. Enregistrer ces éléments avec leurs métadonnées (fichier source, page et,
   selon le cas, numéro du tableau ou de l'image).
3. Créer des représentations vectorielles du texte et des tableaux avec
   **BGE-M3**, puis les stocker dans **ChromaDB** pour effectuer une recherche
   sémantique.
4. Créer séparément des représentations vectorielles des images avec **CLIP**
   et les stocker dans une autre collection ChromaDB.
5. Rechercher les passages textuels les plus proches d'une question et
   afficher leur score, leur page et leur contenu.

La recherche textuelle est une recherche par similarité vectorielle, pas une
conversation générée par un LLM : le script `query.py` affiche les passages
retrouvés sans rédiger de réponse synthétique. Le calcul des vecteurs d'images
est également séparé ; aucun script de recherche d'images à partir d'une
question n'est encore fourni.

## Contenu du dépôt

| Chemin | Rôle |
|---|---|
| `main.py` | Point d'entrée minimal actuel ; affiche un message de bienvenue. |
| `src/extraction.py` | Extrait le texte, les tableaux et les images d'un PDF et crée un manifeste JSON. |
| `src/embeddings.py` | Petit test de chargement de BGE-M3 et de calcul d'un embedding ; n'indexe pas de documents. |
| `src/chroma.py` | Prépare les documents textuels et tabulaires, les découpe et crée l'index ChromaDB. |
| `src/image_embeddings.py` | Calcule et enregistre les embeddings des images dans une collection distincte. |
| `src/query.py` | Exécute une recherche sémantique textuelle dans l'index existant. |
| `src/verify_db.py` | Affiche le nombre de vecteurs et un exemple de métadonnées des collections. |
| `pyproject.toml` | Métadonnées du projet, version minimale de Python et dépendances. |
| `.python-version` | Version Python indiquée pour le projet : `3.13`. |
| `Data/` | PDFs d'entrée, fichiers extraits et base ChromaDB déjà présents dans l'arborescence. |

Deux PDF figurent actuellement dans `Data/` : `maintenance_machine.pdf` et
`maintenance_moteurs.pdf`. Les scripts d'extraction et d'indexation décrits
dans ce README ciblent toutefois uniquement `maintenance_machine.pdf`.
`maintenance_moteurs.pdf` n'est pas encore branché dans ce pipeline.

## Technologies

- **Python 3.13 ou supérieur** (exigence définie par `pyproject.toml`).
- **uv** pour synchroniser l'environnement à partir de `pyproject.toml` et
  `uv.lock`.
- **PyMuPDF** et **LlamaIndex SimpleDirectoryReader** pour lire le PDF et en
  extraire les contenus.
- **LlamaIndex** pour représenter les documents, les découper et construire
  l'index.
- **Hugging Face / Sentence Transformers** avec le modèle `BAAI/bge-m3` pour
  les embeddings textuels.
- **ChromaDB** pour persister les vecteurs localement.
- **Transformers**, **PyTorch** et **Pillow** avec `openai/clip-vit-base-patch32`
  pour traiter les images.

Les modèles Hugging Face peuvent nécessiter un téléchargement lors de leur
première utilisation. Le modèle CLIP utilise MPS lorsqu'il est disponible,
sinon le CPU.

Les dépendances déclarées dans `pyproject.toml` incluent aussi `pandas` et
`pypdf`, qui ne sont pas utilisés directement par les scripts actuels. À
l'inverse, `src/image_embeddings.py` importe `torch` explicitement ; même si
PyTorch est souvent installé comme dépendance indirecte de la pile ML, il
n'apparaît pas comme dépendance directe dans `pyproject.toml`.

## Installation

Depuis la racine du dépôt :

```bash
uv sync
```

Cette commande installe les dépendances déclarées et verrouillées. Les
commandes suivantes utilisent `uv run` afin d'exécuter chaque script dans
l'environnement du projet.

## Pipeline de traitement

Les scripts ne sont pas orchestrés automatiquement par `main.py`. Pour créer
les données et index, lancer les étapes suivantes depuis la racine, dans cet
ordre.

### 1. Extraire le PDF

```bash
uv run python src/extraction.py
```

Le script lit `maintenance_machine.pdf` et :

- crée les dossiers `original/`, `text/`, `tables/`, `images/` et `processed/`
  sous `Data/documents/maintenance_machine/` si nécessaire ;
- copie le PDF dans `original/` uniquement si cette copie n'existe pas déjà ;
- utilise `SimpleDirectoryReader` pour enregistrer le texte non vide de chaque
  page dans un fichier JSON individuel ;
- utilise PyMuPDF pour détecter les tableaux et extraire les images intégrées ;
- crée `processed/extracted_data.json`, un manifeste rassemblant les textes,
  tableaux et références aux images ;
- affiche un récapitulatif du nombre d'éléments extraits.

Chaque élément du manifeste a un type (`text`, `table` ou `image`), un contenu
et des métadonnées. Les métadonnées comprennent notamment la source et le
numéro de page. Les tableaux ont aussi un numéro de tableau ; les images ont un
numéro d'image.

Les tableaux sont convertis en texte, avec les cellules d'une ligne séparées
par ` | `. Pour les images, le manifeste contient un chemin relatif dans le
dossier du document ; les fichiers image eux-mêmes sont enregistrés dans
`images/`.

### 2. Vérifier le modèle d'embeddings (facultatif)

```bash
uv run python src/embeddings.py
```

Ce script charge `BAAI/bge-m3`, encode une phrase de test et affiche la
dimension et quelques valeurs du vecteur. Il ne lit pas le manifeste et ne
modifie pas ChromaDB ; c'est un test isolé du modèle.

### 3. Construire l'index textuel

```bash
uv run python src/chroma.py
```

Le script :

1. charge `processed/extracted_data.json` ;
2. conserve les éléments `text` et `table` non vides et ignore les éléments
   `image` ;
3. associe leurs métadonnées à des documents LlamaIndex ;
4. charge le modèle `BAAI/bge-m3` ;
5. découpe les contenus avec `SentenceSplitter` (500 unités par morceau et
   recouvrement de 80) ;
6. crée une collection persistante nommée `maintenance_pacific` dans
   `Data/chroma_db/` et y enregistre les vecteurs.

**Attention :** à chaque exécution, ce script supprime d'abord la collection
`maintenance_pacific` existante, puis la reconstruit. Les autres collections
de la base ne sont pas supprimées par ce script.

### 4. Construire l'index des images

```bash
uv run python src/image_embeddings.py
```

Ce script lit les éléments `image` du manifeste, charge les images en RGB et
calcule un embedding avec `openai/clip-vit-base-patch32`. Les vecteurs normalisés
sont enregistrés dans la collection `maintenance_images`, avec la page, le
fichier source et le chemin de l'image dans les métadonnées.

L'identifiant de chaque vecteur est construit à partir du nom du dossier, du
numéro de page et du numéro d'image. L'opération `upsert` ajoute ou remplace
ces identifiants. Les erreurs sont affichées et comptées par image, puis le
script continue avec les suivantes.

Cette étape est indépendante de l'indexation textuelle. Elle ne fait ni OCR,
ni génération de description d'image, ni recherche multimodale.

### 5. Rechercher dans les textes indexés

```bash
uv run python src/query.py
```

La question d'exemple est actuellement codée directement dans le script :
« Quelles sont les causes possibles si le compresseur ne démarre pas ? ».
L'index recharge `maintenance_pacific`, calcule l'embedding de la question,
puis récupère les trois passages les plus proches (`similarity_top_k=3`).
Pour chaque résultat, le script affiche le score de similarité, la page, le
type de contenu et jusqu'à 1 000 caractères de texte.

Pour poser une autre question, il faut modifier la variable `question` dans
`src/query.py`. Il n'existe pas encore d'interface en ligne de commande ou web.

### 6. Vérifier les collections

```bash
uv run python src/verify_db.py
```

Le script inspecte `maintenance_pacific` et `maintenance_images`, affiche le
nombre de vecteurs et compte les types trouvés dans leurs métadonnées.
Il attend que les deux collections existent : si une collection n'a pas encore
été créée, la vérification correspondante échouera.

### Point d'entrée actuel

```bash
uv run python main.py
```

Cette commande ne lance pas le pipeline : `main.py` affiche uniquement
`Hello from maintidoc!`. Pour traiter les documents, lancer les scripts de
`src/` individuellement comme indiqué plus haut.

## Collections ChromaDB

La base persistante est configurée dans `Data/chroma_db/`.

| Collection | Contenu | Modèle | Utilisation |
|---|---|---|---|
| `maintenance_pacific` | Morceaux de texte et tableaux | `BAAI/bge-m3` | Recherche sémantique textuelle via `src/query.py`. |
| `maintenance_images` | Embeddings d'images et chemins associés | `openai/clip-vit-base-patch32` | Indexation des images ; pas encore de script de recherche associé. |

Le dossier de base contient des fichiers internes ChromaDB. Il ne faut pas
modifier manuellement leur contenu. Pour reconstruire l'index textuel, relancer
`src/chroma.py` ; cela remplace la collection textuelle, pas toute la base.

## Données et formats

La structure produite par `src/extraction.py` est la suivante :

```text
Data/
├── maintenance_machine.pdf
├── maintenance_moteurs.pdf
├── chroma_db/
└── documents/
    └── maintenance_machine/
        ├── original/
        │   └── maintenance_machine.pdf
        ├── text/
        │   └── page_XXX.json
        ├── tables/
        │   └── page_XXX_table_YY.json
        ├── images/
        │   └── page_XXX_image_YY.<extension>
        └── processed/
            └── extracted_data.json
```

Un enregistrement de texte ou de tableau possède cette forme générale :

```json
{
  "type": "text",
  "content": "Contenu extrait...",
  "metadata": {
    "source": "maintenance_machine.pdf",
    "page": 1
  }
}
```

Les enregistrements de type `table` comportent également `table_number`. Ceux
de type `image` comportent `image_number` et utilisent `content` pour désigner
le fichier image relatif au dossier du document.

Les sorties d'extraction et les bases de vecteurs sont des artefacts
reproductibles : le manifeste est réécrit lors d'une nouvelle extraction et
l'index textuel est reconstruit à chaque indexation. La copie du PDF original,
elle, n'est pas remplacée si elle existe déjà.

## Limites et points d'attention

- **Entrée unique :** le code actuel cible `maintenance_machine.pdf` seulement ;
  le manuel `maintenance_moteurs.pdf` n'est pas encore traité.
- **Point d'entrée non intégré :** `main.py` n'enchaîne aucune étape du
  pipeline.
- **Question fixe :** `src/query.py` utilise une question d'exemple codée dans
  le fichier.
- **Pas de réponse générée :** le système retourne des extraits et des scores,
  sans synthèse ou réponse produite par un modèle de langage.
- **Pas de recherche visuelle :** les images sont vectorisées, mais aucun
  script ne recherche les images par texte ou par image.
- **Pas d'OCR dédié :** les images ne sont pas converties en texte et le code
  indique qu'elles sont traitées séparément.
- **Chemins sensibles à la casse :** le code utilise le dossier `data/` en
  minuscules, tandis que le dossier présent dans le dépôt s'appelle `Data/`.
  Cela fonctionne généralement sur un volume macOS configuré sans distinction
  de casse, mais peut échouer sur un système de fichiers sensible à la casse.
  Harmoniser le nom du dossier ou les chemins avant d'exécuter les scripts sur
  ce type de système.
- **Dépendance à l'ordre :** l'extraction doit précéder les scripts
  d'indexation ; `query.py` dépend de l'existence de la collection textuelle,
  et `verify_db.py` attend les deux collections.
- **Pas de tests automatisés présents dans le dépôt :** la validation actuelle
  se fait par l'exécution des scripts et leurs messages de sortie.