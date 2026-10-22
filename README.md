# P13 — Agent IA d'apprentissage des ouvertures aux échecs

Application destinée à la FFE pour apprendre les ouvertures aux jeunes espoirs :
un échiquier interactif (Angular) guidé par un agent IA s'appuyant sur la théorie
des ouvertures, Stockfish, les APIs Lichess, une base vectorielle Milvus et des
vidéos YouTube.

Le détail du besoin est décrit dans [docs/product_backlog.md](docs/product_backlog.md).

## Structure du projet

```
P13/
├── backend/             # API FastAPI (Python 3.13, dépendances gérées par uv)
│   ├── app/             # Code de l'application
│   ├── tests/           # Tests pytest
│   └── Dockerfile
├── frontend/            # Application Angular 21 (voir frontend/README.md)
├── docs/                # Documentation du projet
├── docker-compose.yml   # Orchestration des services
└── .env.example         # Modèle de configuration
```

## Prérequis

- Git
- Docker et Docker Compose
- [uv](https://docs.astral.sh/uv/) (uniquement pour lancer le backend hors Docker)

## Démarrage

```bash
cp .env.example .env            # LICHESS_TOKEN et YOUTUBE_API_KEY facultatifs mais recommandés
ollama pull qwen3:14b           # modèle de l'agent, sur la machine hôte (hors Docker)
docker compose up -d --build    # frontend (nginx), backend (Stockfish), MongoDB, Milvus
docker compose exec backend uv run python -m scripts.ingest_corpus   # corpus → Milvus, une fois
```

L'application est sur <http://localhost:8080> (échiquier, coups des maîtres, conseil de
l'agent, analyse profonde, vidéos, questions). Détails et positions de démo dans
[docs/deploiement.md](docs/deploiement.md).

Vérifier que l'API répond :

```bash
curl http://localhost:8001/api/v1/healthcheck
# {"status":"ok","environment":"development"}

curl http://localhost:8001/api/v1/healthcheck/database
# {"status":"ok","database":"p13"}
```

La documentation interactive de l'API est disponible sur <http://localhost:8001/docs>.

Arrêter les services :

```bash
docker compose down
```

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
| [docs/processus.md](docs/processus.md) | Processus, étapes, graphe de l'agent |
| [docs/contrat-tools.md](docs/contrat-tools.md) | Les sept outils de l'agent et leur contrat |
| [docs/outil-rag.md](docs/outil-rag.md) | RAG agentique, phases 0 à 7 et mesures |
| [docs/outil-videos.md](docs/outil-videos.md) | Recherche de vidéos YouTube, quotas, mesure |
| [docs/frontend.md](docs/frontend.md) | Front Angular : services, panneau, synchronisation |
| [docs/deploiement.md](docs/deploiement.md) | Conteneurs, volumes, installation fraîche, démo |

## Services

| Service  | Technologie | État    |
|----------|-------------|---------|
| frontend | Angular 21 servi par nginx | En place (port 8080) |
| backend  | FastAPI, LangGraph, Stockfish 17.1 | En place (port 8001) |
| milvus   | Milvus 2.6  | En place (base vectorielle du RAG) |
| mongodb  | MongoDB 8.0 | En place (corpus, cache vidéos, quota) |
| langfuse | Langfuse v4 (profil `tracing`, optionnel) | Arrêté par défaut |
