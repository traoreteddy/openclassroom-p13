# P13 — Agent IA d'apprentissage des ouvertures aux échecs

Application destinée à la FFE pour apprendre les ouvertures aux jeunes espoirs :
un échiquier interactif (Angular) guidé par un agent IA s'appuyant sur la théorie
des ouvertures, Stockfish, les APIs Lichess, une base vectorielle Milvus et des
vidéos YouTube.

## Structure du projet

```
P13/
├── backend/             # API FastAPI (Python 3.13, dépendances gérées par uv)
│   ├── app/             # Code de l'application
│   ├── tests/           # Tests pytest
│   └── Dockerfile
├── frontend/            # Application Angular 21 (voir frontend/README.md)
├── docs/                # Note sur les bénéfices et limites du système (PDF)
├── docker-compose.yml   # Orchestration des services
└── .env.example         # Modèle de configuration
```

## Prérequis

| Outil | Pourquoi |
|---|---|
| Git | récupérer le dépôt |
| Docker Desktop (ou Docker Engine + Compose v2), **8 Go de RAM** alloués | tous les services tournent en conteneurs |
| [Ollama](https://ollama.com) sur la machine hôte | le modèle de langage de l'agent (hors Docker, pour utiliser le GPU du Mac) |
| 6 Go d'espace disque | images, modèle d'embedding (1,2 Go), modèle Ollama (9 Go pour qwen3:14b) |
| Facultatif : jeton Lichess, clé YouTube Data v3 | coups des maîtres sans limite de débit ; vidéos |
| Facultatif : [uv](https://docs.astral.sh/uv/), Node 22 | backend et front hors Docker (développement) |

## Installation et démarrage

```bash
git clone https://github.com/traoreteddy/openclassroom-p13.git P13 && cd P13
cp .env.example .env
```

Dans `.env`, renseigner si vous les avez `LICHESS_TOKEN` et `YOUTUBE_API_KEY` ; les
autres valeurs par défaut conviennent pour une démonstration locale.

```bash
ollama pull qwen3:14b                 # modèle de l'agent (une fois, ≈ 9 Go)
ollama pull phi4-mini                 # résumés d'enrichissement du corpus (une fois)
docker compose up -d --build          # frontend, backend, MongoDB, Milvus (+ etcd, minio)
docker compose ps                     # attendre que tout soit « healthy » (1 à 3 min)
docker compose exec backend uv run python -m scripts.ingest_corpus   # corpus → Milvus (une fois)
```

Ouvrir <http://localhost:8080>. L'API est aussi exposée sur <http://localhost:8001>
(documentation interactive sur `/docs`).

### Vérifier que tout est interfacé

```bash
./scripts/smoke.sh
```

Le script appelle chaque service **à travers le front** : page, backend, MongoDB, Lichess,
Stockfish (évaluation et analyse différée), Milvus, YouTube, puis l'agent (conseil complet
et question libre). Il affiche une ligne par vérification et sort en erreur si l'une échoue.

### Positions pour la démonstration

| Scénario | Comment l'obtenir | Ce que l'on voit |
|---|---|---|
| En théorie | jouer 1.e4 e5 2.Cf3 Cc6 3.Fc4, ou charger la FEN `r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3` | « Italian Game · C50 », Fc5 25 k parties, vidéos françaises, conseil avec évaluation et extraits de l'encyclopédie |
| Sicilienne | 1.e4 c5 2.Cf3 d6 3.d4 | statistiques, nom ECO, vidéos sur la Sicilienne |
| Hors théorie | 1.a4 h5 2.Ta3 Th6 3.Tg3 | badge « Hors théorie », conseil fondé sur Stockfish, pas de vidéo |
| Position invalide | FEN `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0` | message « FEN invalide », aucun appel à l'API |
| Analyse profonde | bouton « Analyse profonde » | suivi de la tâche de fond, résultat à la profondeur 28 |
| Question libre | « Quelles sont les idées de la Caro-Kann ? » | réponse rédigée à partir de l'encyclopédie |

### Arrêter, repartir, nettoyer

```bash
docker compose down                   # arrête et supprime les conteneurs, garde les volumes
docker compose up -d                  # repart avec les données (corpus, index, cache)
docker compose down -v                # supprime aussi les volumes (réinstallation complète)
docker compose --profile tracing up -d   # facultatif : Langfuse sur http://localhost:3000
```

### En cas de problème

| Symptôme | Cause probable | Remède |
|---|---|---|
| `backend` reste « starting » | Milvus ou MongoDB pas encore sains, ou machine chargée | `docker compose ps`, attendre ; `docker compose logs backend` |
| Carte « Coups des maîtres » en erreur 503 | Lichess injoignable ou limite de débit sans jeton | renseigner `LICHESS_TOKEN` |
| Conseil de l'agent en erreur 500 | Ollama absent, modèle non téléchargé, ou `OLLAMA_BASE_URL` renseigné avec `localhost` dans `.env` (dans un conteneur, `localhost` est le conteneur) | `ollama list`, `ollama pull qwen3:14b` ; laisser `OLLAMA_BASE_URL` vide ; sous Linux, vérifier `host.docker.internal` |
| Ingestion très lente dans le conteneur | encodeur en bfloat16 sur CPU | laisser `EMBEDDING_DTYPE` vide (le compose impose float32) |
| « Vidéos : non configurée » | pas de clé YouTube | renseigner `YOUTUBE_API_KEY` |
| Recherche vectorielle `not_indexed` | corpus non ingéré | lancer `scripts.ingest_corpus` |
| Port déjà utilisé | un autre service écoute sur 8080, 8001, 27017 ou 19530 | changer `FRONTEND_PORT`, `BACKEND_PORT`, `MONGO_PORT`, `MILVUS_PORT` dans `.env` |

## Configuration

La configuration passe par des variables d'environnement définies dans `.env`.

| Variable        | Défaut                   | Description                              |
|-----------------|--------------------------|------------------------------------------|
| `APP_NAME`      | `P13 Chess Openings API` | Nom affiché de l'API                     |
| `APP_ENV`       | `development`            | Environnement d'exécution                |
| `API_V1_PREFIX` | `/api/v1`                | Préfixe des routes de l'API              |
| `BACKEND_PORT`  | `8001`                   | Port de l'API exposé sur la machine hôte |
| `FRONTEND_PORT` | `8080`                   | Port du front (nginx, relaie `/api` vers le backend) |
| `YOUTUBE_API_KEY` | — | Clé YouTube Data v3 pour les vidéos (sinon carte « non configurée ») |
| `MONGO_ROOT_USERNAME` | `p13`              | Utilisateur administrateur MongoDB       |
| `MONGO_ROOT_PASSWORD` | `change-me`        | Mot de passe MongoDB (à changer)         |
| `MONGODB_DB_NAME` | `p13`                  | Nom de la base de données                |
| `MONGO_PORT`    | `27017`                  | Port MongoDB exposé sur la machine hôte  |
| `MONGODB_URL`   | `mongodb://p13:change-me@localhost:27017` | URL utilisée par l'API hors Docker |
| `LICHESS_TOKEN` | — | Jeton Lichess, **obligatoire** pour l'Opening Explorer |
| `THEORY_MIN_GAMES` | `20` | Parties minimum pour qu'une position soit « en théorie » |
| `STOCKFISH_PATH` | `stockfish` | Exécutable Stockfish (`brew install stockfish` en local) |
| `YOUTUBE_API_KEY` | — | Clé API YouTube Data v3 |

## Développement du backend hors Docker

MongoDB doit être démarré (`docker compose up -d mongodb`).

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8001
```

Lancer les tests (sans réseau, avec couverture de branches) :

```bash
cd backend
uv run pytest --cov
```

Documentation :

| Document | Contenu |
|---|---|
| [docs/architecture_bef_limit.pdf](docs/architecture_bef_limit.pdf) | Note détaillée sur les bénéfices et limites du système : schéma d'architecture technique, étude de faisabilité et estimation des coûts |

## Services

| Service  | Technologie | État    |
|----------|-------------|---------|
| frontend | Angular 21 servi par nginx | En place (port 8080) |
| backend  | FastAPI, LangGraph, Stockfish 17.1 | En place (port 8001) |
| milvus   | Milvus 2.6  | En place (base vectorielle du RAG) |
| mongodb  | MongoDB 8.0 | En place (corpus, cache vidéos, quota) |
| langfuse | Langfuse v4 (profil `tracing`, optionnel) | Arrêté par défaut |
