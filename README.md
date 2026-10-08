# loom-notes

Une mémoire locale pour Claude Desktop et les agents IA. Serveur MCP + RAG hybride (BGE-M3, reranking, Qdrant). L'utilisateur y met ce qu'il veut, Claude vient y chercher. Tout tourne en local, sans cloud ni clé API.

## Pourquoi

La mémoire automatique des assistants enregistre ce qu'elle croit avoir compris : des demi-vérités, des choses périmées, des inférences. Ici, rien n'entre sans que l'utilisateur l'ait décidé. Il ajoute un texte, une page web ou un fichier markdown ; Claude interroge cette base avant de répondre sur ses projets, ses décisions ou ses notes, et n'écrit dedans que si l'utilisateur le lui demande explicitement dans le message courant.

Le contenu vit dans un dossier local (`~/.local/share/loom-notes` par défaut). Les modèles d'embedding et de reranking tournent en local. Le réseau ne sert qu'à deux choses : télécharger les modèles au premier usage, et télécharger une page quand l'utilisateur demande de la mémoriser.

Le même serveur sert Claude Desktop (chat et Cowork) et tout agent qui parle MCP : les tools sont les mêmes.

## Ce que Claude peut faire

Neuf tools, séparés en deux familles.

| Tool | Rôle |
|---|---|
| `search(query, project?, tags?, k)` | Recherche hybride puis reranking. Sans `project`, la recherche est globale et chaque résultat indique son projet. `k` est plafonné à 10. |
| `get(doc_id)` | Texte intégral d'un document et ses métadonnées. |
| `list_docs(project?, n)` | Derniers documents ajoutés, du plus récent au plus ancien, sans le texte. |
| `projects()` | Projets présents et nombre de documents pour chacun. |
| `add_text(text, title, project, tags?)` | Ajoute un texte brut. |
| `add_url(url, project, tags?)` | Télécharge une page, en extrait le contenu et l'ajoute. Le titre est celui de la page. |
| `add_file(path, project, tags?)` | Ajoute un fichier markdown local, découpé par titres. Soumis au contrôle d'accès décrit plus bas. |
| `update(doc_id, text)` | Remplace le texte d'un document. Même identifiant, titre, projet et tags conservés. |
| `delete(doc_id)` | Supprime définitivement un document et ses chunks. |

Les quatre premiers sont déclarés en lecture seule au sens MCP. Les cinq autres portent, dans leur description et dans les instructions du serveur, la même consigne : n'appeler que sur demande explicite de l'utilisateur, jamais de sa propre initiative. C'est une consigne au modèle, pas un contrôle technique. Le seul contrôle technique porte sur `add_file`.

Chaque tool d'écriture renvoie ce qu'il a réellement fait : identifiant, titre, projet, nombre de chunks écrits, et le cas échéant `duplicate_of` (contenu identique à un document existant, rien n'a été écrit) ou `updated` (un document existant a été remplacé).

`search` renvoie le chunk entier quand il fait moins de 1 000 caractères, ce qui couvre la plupart des notes et évite de couper un tableau au milieu. Au delà, un extrait de 300 caractères et `truncated: true`, signal pour appeler `get`.

## Comment ça marche

### Ingestion

Les trois entrées convergent vers le même pipeline. Le texte brut est pris tel quel. Une page web est convertie en markdown structuré : titres, listes, tableaux et blocs de code sont conservés, la navigation, le pied de page, les scripts et les formulaires retirés. Un fichier markdown est lu tel quel, son titre est le premier `# H1` ou, à défaut, le nom du fichier.

Le texte est ensuite découpé par sections, en suivant les titres markdown. Une section trop longue est retaillée en paragraphes à environ 1 800 caractères avec un chevauchement de 180. Chaque chunk est préfixé par son chemin de titres (`Titre du document > Section > Sous-section`) avant d'être vectorisé : un paragraphe isolé perd son contexte, et c'est le détail qui fait le plus de différence en rappel sur des notes.

Une empreinte SHA-256 du texte normalisé (casse, espaces) sert de déduplication. Un contenu déjà présent n'est pas réécrit. Une source déjà connue, URL ou chemin de fichier, dont le contenu a changé remplace le document existant au lieu d'en créer un second.

BGE-M3 produit en un seul passage un vecteur dense (1 024 dimensions) et un vecteur sparse lexical par chunk.

### Retrieval

La requête est vectorisée de la même façon. Qdrant exécute deux recherches, dense et sparse, avec les filtres de projet et de tags appliqués dans chacune, puis fusionne les deux listes par RRF. Les candidats passent ensuite dans le cross-encoder `bge-reranker-v2-m3`, qui note chaque couple question/chunk entre 0 et 1. Les résultats sont regroupés à deux chunks maximum par document, filtrés par un score minimal, puis tronqués à `k`.

Le score renvoyé est celui du reranker. Au dessus de 0,8, le passage répond directement à la question ; entre 0,3 et 0,8, il traite le sujet sans y répondre franchement ; en dessous de 0,1, c'est du bruit ramené par le vecteur faute de mieux. Le seuil par défaut est 0,1 ; la section Évaluation explique comment le régler.

### Stockage

Deux collections Qdrant. `documents` contient le texte intégral et les métadonnées, sans vecteur : c'est la source de vérité. `memory` contient les chunks vectorisés et se reconstruit entièrement depuis `documents` avec `loom-notes reindex`. Les identifiants de chunk sont dérivés de l'identifiant du document et de l'index du chunk, donc une réécriture ne laisse pas d'orphelin.

Le nom du modèle d'embedding est enregistré dans `DATA_DIR/meta.json` au moment de l'indexation. Si la configuration demande un autre modèle, le serveur refuse de démarrer et demande un `reindex` : deux modèles ne se mélangent jamais en silence dans le même index.

## Installation

Linux, Python 3.12 ou plus récent, et une machine capable de faire tourner deux modèles de 570 M de paramètres. Les modèles sont dans l'extra `models` : sans lui, le serveur démarre, mais la première recherche échoue en demandant de l'installer.

Avec [uv](https://docs.astral.sh/uv/), en outil isolé :

```bash
uv tool install "loom-notes[models]"
```

Les commandes `loom-notes` et `loom-notes-mcp` arrivent dans `~/.local/bin`. Avec pip, dans un environnement virtuel : `pip install "loom-notes[models]"`.

Sous Linux, la roue torch de PyPI embarque CUDA et pèse plusieurs Go. Sur une machine sans GPU NVIDIA, la version CPU, bien plus légère, s'installe d'abord depuis l'index de PyTorch, dans le même environnement :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "loom-notes[models]"
```

Avec uv, le même enchaînement fonctionne dans un environnement créé par `uv venv`, avec `uv pip install` à la place de `pip install`.

Les poids de BGE-M3 (environ 3 Go) et du reranker (environ 2,3 Go) sont téléchargés depuis Hugging Face au premier usage et mis en cache dans `~/.cache/huggingface` (`HF_HOME` pour le déplacer). Il n'y a rien d'autre à télécharger ensuite.

Sur GPU, les deux modèles tiennent dans 3 Go de VRAM en fp16. Le périphérique est choisi au chargement : cuda, sinon mps, sinon CPU ; `LOOM_NOTES_DEVICE` l'impose, et le fp16 n'est activé que sur cuda. Sur CPU, le reranker devient le poste dominant : comptez une dizaine de secondes par recherche. Une carte Pascal ou plus ancienne (compute capability inférieure à 7.5) n'est pas prise en charge par les roues torch CUDA 13 publiées sur PyPI ; il faut alors soit le CPU, soit une roue CUDA 12.6.

Le pinning `transformers<5` n'est pas un oubli : FlagEmbedding 1.4 casse avec transformers 5 (`tokenizer.pad` reçoit une liste au lieu d'un dictionnaire).

## Qdrant

Sans `LOOM_NOTES_QDRANT_URL`, Qdrant tourne embarqué dans `DATA_DIR/qdrant` (`QdrantClient(path=…)`) : rien à installer, mais un seul processus à la fois. Ce mode suffit pour la CLI et les tests. Il ne convient pas à Claude Desktop, qui lance deux instances de chaque serveur MCP, une pour le chat et une pour les sessions Cowork et Code : la seconde mourrait à l'ouverture. Pour Claude Desktop, il faut un Qdrant en conteneur :

```bash
mkdir -p /home/[user]/.local/share/qdrant
docker run -d --name qdrant --restart unless-stopped \
  --user "$(id -u):$(id -g)" \
  -e QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/storage/snapshots \
  -p 127.0.0.1:6333:6333 \
  -v /home/[user]/.local/share/qdrant:/qdrant/storage \
  qdrant/qdrant
curl -s localhost:6333/
```

Le dossier du volume est créé avant le conteneur : sinon Docker le crée au nom de root, et Qdrant, lancé sous l'utilisateur, ne peut pas y écrire. Il est distinct de `DATA_DIR/qdrant`, le chemin du mode embarqué.

`--user` fait que les fichiers du volume appartiennent à l'utilisateur et non à root ; la variable `QDRANT__STORAGE__SNAPSHOTS_PATH` est nécessaire dans ce cas, sinon Qdrant ne peut pas écrire son dossier de snapshots dans l'image. Le port n'est exposé que sur l'interface locale.

`--restart unless-stopped` relance le conteneur avec le service Docker au démarrage de la machine, à condition que ce service soit activé (`systemctl is-enabled docker`, sinon `sudo systemctl enable docker`). Un `docker stop qdrant` manuel le laisse arrêté jusqu'au prochain `docker start qdrant`.

Les deux modes n'ont pas le même format sur disque, on ne passe pas de l'un à l'autre sans réingérer (`export` puis `import`).

## Branchement dans Claude Desktop

`~/.config/Claude/claude_desktop_config.json` :

```json
{
  "mcpServers": {
    "loom-notes": {
      "command": "/home/[user]/.local/bin/loom-notes-mcp",
      "args": [],
      "env": {
        "FASTMCP_SHOW_SERVER_BANNER": "false",
        "FASTMCP_CHECK_FOR_UPDATES": "off",
        "LOOM_NOTES_QDRANT_URL": "http://127.0.0.1:6333",
        "LOOM_NOTES_ALLOWED_ROOTS": "/home/[user]/dev"
      }
    }
  }
}
```

`command` est le chemin absolu du binaire installé, que donne `which loom-notes-mcp`. Pas de `uvx` ni de `uv run` ici : au premier lancement, ils résoudraient et téléchargeraient les dépendances, torch compris, entre l'`exec()` et le handshake MCP, sans rien écrire sur la sortie standard pendant ce temps, et le client attend. Fermer complètement l'application avant d'éditer ce fichier, elle le réécrit à la fermeture.

Au démarrage, le serveur répond au handshake tout de suite et charge les modèles en tâche de fond, après les avoir téléchargés la toute première fois. La première recherche peut attendre la fin de ce chargement, les suivantes non.

## Données

Tout ce que loom-notes écrit sur le disque est dans `DATA_DIR` : `$XDG_DATA_HOME/loom-notes`, sinon `~/.local/share/loom-notes`, ou le dossier donné par `LOOM_NOTES_DATA_DIR`.

| Fichier | Contenu |
|---|---|
| `meta.json` | Modèle d'embedding avec lequel l'index a été construit. |
| `qdrant/` | La base, en mode embarqué seulement. |
| `export.jsonl` | Sauvegarde écrite par `loom-notes export`. |
| `golden.jsonl` | Jeu doré de `loom-notes eval`. |

En mode serveur, la base est dans le volume du conteneur Qdrant ; `DATA_DIR` ne garde que les trois fichiers. Le serveur et la CLI doivent voir le même `DATA_DIR` et le même `QDRANT_URL`.

## Configuration

Toutes les options sont des variables d'environnement préfixées `LOOM_NOTES_` : celles du bloc `env` de Claude Desktop pour le serveur, celles du shell pour la CLI. Aucun fichier `.env` n'est lu. Les valeurs ci-dessous sont les défauts.

| Variable | Défaut | Rôle |
|---|---|---|
| `DATA_DIR` | `$XDG_DATA_HOME/loom-notes`, sinon `~/.local/share/loom-notes` | Répertoire des données : Qdrant embarqué, `meta.json`, export, jeu doré. |
| `QDRANT_URL` | vide | URL d'un serveur Qdrant. Vide : mode embarqué dans `DATA_DIR/qdrant`. |
| `QDRANT_API_KEY` | vide | Clé d'API si le serveur en exige une. |
| `DEVICE` | `auto` | Périphérique torch. `auto` : cuda, sinon mps, sinon cpu. Le fp16 n'est activé que sur cuda. |
| `DENSE_MODEL` | `BAAI/bge-m3` | Modèle d'embedding. En changer impose un `reindex`. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder de reranking. |
| `ALLOWED_ROOTS` | vide | Racines lisibles par `add_file`, séparées par `:`. Vide : `add_file` refusé. |
| `DENY_PATTERNS` | vide | Motifs refusés en plus de la liste de base, séparés par `:`. |
| `CHUNK_TARGET_CHARS` | 1800 | Taille visée d'un chunk. |
| `CHUNK_MAX_CHARS` | 2400 | Taille maximale d'une section avant retaille. |
| `CHUNK_OVERLAP_CHARS` | 180 | Chevauchement entre chunks consécutifs d'une même section. |
| `CHUNK_MIN_CHARS` | 200 | En dessous, une section est fusionnée avec la précédente. |
| `PREFETCH_LIMIT` | 15 | Candidats par branche (dense, sparse) avant fusion. |
| `RERANK_CANDIDATES` | 20 | Candidats passés au reranker. |
| `MAX_CHUNKS_PER_DOC` | 2 | Chunks d'un même document dans les résultats. |
| `FULL_CHUNK_CHARS` | 1000 | En dessous, `search` renvoie le chunk entier. |
| `SNIPPET_CHARS` | 300 | Longueur de l'extrait au delà. |
| `MIN_SCORE` | 0.1 | Score reranker minimal d'un résultat. |
| `WARMUP_ON_START` | `true` | Préchargement des modèles au démarrage du serveur. |
| `FAKE_MODELS` | `false` | Modèles factices, pour les tests ou une démo sans GPU. |
| `FETCH_TIMEOUT_S` | 20 | Délai de téléchargement pour `add_url`. |
| `FETCH_MAX_BYTES` | 5000000 | Taille maximale d'une page téléchargée par `add_url`, après décompression. |
| `FETCH_ALLOW_PRIVATE` | `false` | `true` autorise `add_url` vers la machine et le réseau local. |

## Sécurité de `add_file`

Un serveur MCP tourne avec les droits de l'utilisateur qui lance Claude Desktop, et Claude Desktop ne lui applique aucune restriction de dossier. Sans garde-fou, `add_file` suivi de `get` est une lecture de fichier arbitraire, y compris depuis une session Cowork dont les dossiers autorisés sont bien plus étroits. Ce n'est pas théorique, c'est le test qui a motivé ce paragraphe.

Trois contrôles s'appliquent, dans cet ordre, et le message d'erreur nomme celui qui a mordu.

Le chemin résolu, symlinks compris, doit être inclus dans une des racines de `ALLOWED_ROOTS`, elles-mêmes résolues. Un lien qui sort d'une racine est refusé. Sans racine configurée, `add_file` est refusé d'office.

Aucun composant du chemin relatif à la racine ne doit correspondre à un motif refusé. La liste de base est dans le code et ne peut pas être retirée par configuration :

```
.env  .env.*  *.pem  *.key  *.p12  *.pfx
id_rsa*  id_ecdsa*  id_ed25519*
.ssh  .aws  .gnupg  .netrc  .npmrc  .pypirc
.git  .venv  node_modules  __pycache__
```

`DENY_PATTERNS` peut seulement y ajouter. Une garantie qu'un fichier de configuration peut désactiver n'en est plus une.

L'extension doit être `.md`, `.markdown` ou `.txt`.

Ce filtrage porte sur des noms. Un secret écrit en clair dans une note markdown d'une racine autorisée sera lu. La frontière, c'est le choix des racines ; le reste est de la défense en profondeur. `add_url` a ses propres contrôles, décrits ci-dessous, et `add_text` ne lit rien.

## Sécurité de `add_url`

`add_url` est la seule sortie réseau que le modèle peut déclencher, et une page mémorisée peut contenir une injection. Deux contrôles s'appliquent à chaque requête, redirections comprises.

L'hôte doit être public. Une adresse IP écrite dans l'URL, ou chacune des adresses renvoyées par le DNS, est vérifiée : boucle locale, réseaux privés, lien local, adresses réservées et multicast sont refusés. Sans ce contrôle, une page pourrait faire lire au serveur `http://127.0.0.1:6333/collections` (l'API de Qdrant) ou l'interface d'un équipement du réseau local, et la ranger en mémoire. `FETCH_ALLOW_PRIVATE=true` lève ce contrôle, pour mémoriser la page d'un service interne.

La page ne doit pas dépasser `FETCH_MAX_BYTES` (5 Mo par défaut), mesurés après décompression : une réponse annoncée plus grosse est refusée avant le téléchargement, une réponse qui dépasse en cours de route est coupée.

Seules les URL http et https sont acceptées, et seul le contenu HTML ou XML.

## Ligne de commande

`loom-notes` expose les mêmes opérations que le serveur, plus la maintenance. En mode serveur Qdrant, la CLI fonctionne pendant que Claude Desktop tourne. Elle lit les mêmes variables que le serveur, à exporter dans le shell : au minimum `LOOM_NOTES_QDRANT_URL`, et `LOOM_NOTES_DATA_DIR` si le dossier de données n'est pas celui par défaut.

```
loom-notes add-text PROJET TITRE [TEXTE]      texte brut ; lu sur stdin si absent
loom-notes add-url PROJET URL
loom-notes add-file PROJET CHEMIN
loom-notes search "question" [-p projet] [-t tag] [-k 5]
loom-notes get DOC_ID
loom-notes list [-p projet] [-n 20]
loom-notes projects
loom-notes delete DOC_ID
loom-notes export [FICHIER]                    sauvegarde JSONL, défaut DATA_DIR/export.jsonl
loom-notes import [FICHIER]                    réimport ; les doc_id déjà présents sont ignorés
loom-notes reindex                             reconstruit les chunks depuis les documents
loom-notes eval [FICHIER] [--json]             évalue le retrieval sur le jeu doré
loom-notes eval-add "question" DOC_ID [-p projet]
```

Les commandes d'ajout acceptent `-t` plusieurs fois pour les tags. Les options globales `--fake` (modèles factices) et `--data-dir` se placent avant la commande.

## Évaluation

`DATA_DIR/golden.jsonl` est un jeu doré : une ligne par cas, avec la question telle que l'utilisateur la poserait et l'identifiant du document qui doit sortir.

```json
{"query":"comment purger le journal d'audit sans arrêter le serveur","doc_id":"f3c9d186-…","title":"loom-fs — serveur MCP filesystem à rôles"}
```

`loom-notes eval` rejoue chaque question à seuil zéro et donne, par cas, le rang du document attendu et son score, puis recall@1, recall@5 et MRR, et enfin le plus haut seuil qui ne fait perdre aucun cas trouvé, avec la part des résultats hors document attendu qui tomberaient sous ce seuil. C'est la valeur à mettre dans `MIN_SCORE`.

Le jeu s'enrichit au fil de l'eau : deux questions par document ajouté, avec `eval-add`. Un cas qui sort avec un score faible alors que le document contient la réponse signale presque toujours un document mal rédigé, typiquement une commande sans la phrase qui dit quand et pourquoi l'utiliser. Sur les premiers cas, la même question passait de 0,91 à 0,11 selon que le document gardait ou non ses deux phrases de contexte.

Un changement de chunking, de modèle, de seuil ou de fusion se valide par un `eval` avant et après.

## Sauvegarde et réindexation

`loom-notes export` écrit un document par ligne dans `DATA_DIR/export.jsonl`, texte intégral et métadonnées compris. C'est la sauvegarde de référence, indépendante de Qdrant et du modèle d'embedding.

`loom-notes import` recharge un export dans une base vide ou partielle, en réindexant chaque document et en ignorant les identifiants déjà présents. Changer de modèle d'embedding ou de mode Qdrant revient à un `export`, un changement de configuration, puis un `import` ou un `reindex`.

## Dépannage

**Le serveur affiche Échec dans Claude Desktop avec « Connection closed ».** Lire les journaux depuis les paramètres de l'application. La première ligne d'erreur du serveur dit ce qui manque. Les deux causes habituelles : Qdrant injoignable (« Qdrant injoignable sur http://127.0.0.1:6333 … `docker start qdrant` »), ou un `command` qui ne désigne pas le binaire : y mettre le chemin absolu donné par `which loom-notes-mcp`.

**Le serveur refuse de démarrer en parlant de modèle.** `DATA_DIR/meta.json` enregistre le modèle avec lequel l'index a été construit et il diffère de `DENSE_MODEL`. `loom-notes reindex` reconstruit l'index avec le modèle courant.

**`add_file` est refusé.** Le message nomme la règle : hors des racines autorisées, motif refusé, extension. Sans `ALLOWED_ROOTS`, tout est refusé.

**`add_url` est refusé.** « adresse non publique refusée » : la page est sur la machine ou le réseau local ; `LOOM_NOTES_FETCH_ALLOW_PRIVATE=true` si c'est voulu. « page trop volumineuse » : relever `LOOM_NOTES_FETCH_MAX_BYTES`.

**Une recherche prend dix secondes.** Le reranker tourne sur CPU. Vérifier `LOOM_NOTES_DEVICE` et que torch voit bien la carte (`python -c "import torch; print(torch.cuda.is_available())"` avec le Python de l'installation, `~/.local/share/uv/tools/loom-notes/bin/python` pour une installation par `uv tool`).

**`IndexError: list index out of range` dans FlagEmbedding au premier appel.** Le GPU est vu par torch mais sans kernels compatibles (carte trop ancienne pour la roue CUDA installée). FlagEmbedding attrape l'erreur CUDA, réduit son batch jusqu'à zéro et plante sur une liste vide. Passer en `cpu` ou installer une roue torch adaptée à la carte.

## Limites connues

Le contrôle « écriture sur demande explicite » est une consigne au modèle. Une page ajoutée par `add_url` peut contenir une injection ; elle n'aura aucun pouvoir sur le disque grâce au confinement de `add_file`, mais elle pourrait pousser le modèle à écrire dans la mémoire. Les tools d'écriture renvoient toujours ce qu'ils ont fait, c'est à l'utilisateur de le lire.

Le contrôle d'adresse de `add_url` résout le nom, puis httpx se connecte : un DNS qui change de réponse entre les deux (DNS rebinding) passerait au travers. Il protège des liens et des redirections vers le réseau local, pas d'un serveur DNS hostile.

Les tailles de chunk sont en caractères, avec l'approximation de quatre caractères par token en français, pas en tokens du modèle.

`list_docs` et `projects` parcourent toute la collection `documents`. Linéaire, invisible jusqu'à quelques milliers de documents.

Le mode serveur Qdrant ne chiffre rien et n'authentifie personne par défaut. Le port est lié à l'interface locale ; ne pas l'exposer sans clé d'API.

Il n'y a pas de quota. Un agent en boucle peut lancer autant de recherches qu'il veut ; chacune coûte du temps de reranking, pas d'argent.

## Développement

```bash
git clone https://github.com/denislamard/loom-notes.git
cd loom-notes
uv sync
```

`uv sync` installe aussi l'extra `models` (FlagEmbedding, torch), par le groupe du même nom. Les tests n'en ont pas besoin : `uv sync --no-group models` s'en passe.

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pyright
uv run pytest
```

Les tests tournent sans GPU et sans téléchargement : les modèles sont remplacés par des factices déterministes (dense par sac de mots haché, sparse par comptage) suffisants pour vérifier le filtrage, la déduplication, le remplacement par source, l'export et l'import, le confinement de `add_file`, les contrôles de `add_url` et le comportement des tools à travers un client MCP en mémoire. Pyright est en mode strict.

La CI (`.github/workflows/ci.yml`) lance ces contrôles sur Python 3.12, 3.13 et 3.14, puis construit le paquet et vérifie son contenu. `.github/workflows/publish.yml` publie : à la main vers TestPyPI, et vers PyPI à chaque étiquette `v*` dont le numéro est celui du `pyproject.toml`.

## Licence

Apache 2.0.
