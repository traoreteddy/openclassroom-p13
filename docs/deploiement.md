# Déploiement : `docker compose` complet, volumes et démo

Étape 6 du processus (`docs/processus.md`). Tout tourne en conteneurs sur la machine,
sauf Ollama (sur l'hôte, pour le GPU du Mac) : voir la décision en `docs/outil-rag.md` § 1.

## 1. Services

| Service | Image | Rôle | Port hôte | Santé |
|---|---|---|---|---|
| `frontend` | `nginx:1.27-alpine` (build Angular dans `node:22-alpine`) | Application Angular ; relaie `/api/` vers le backend (une seule origine, pas de CORS) ; délai de 300 s pour le conseil de l'agent | `FRONTEND_PORT` = 8080 | `wget` sur `/` |
| `backend` | `backend/Dockerfile` (Python 3.13, uv, **Stockfish 17.1**) | API FastAPI, agent LangGraph, outils | `BACKEND_PORT` = 8001 | `/api/v1/healthcheck` |
| `mongodb` | `mongo:8` | Articles bruts, cache vidéos, compteur de quota YouTube, parties (à venir) | 27017 | `mongosh ping` |
| `milvus` + `milvus-etcd` + `milvus-minio` | `milvusdb/milvus:v2.6.25` | Base vectorielle du RAG | 19530 | `/healthz` |
| `langfuse-*` (profil `tracing`, **optionnel**) | Langfuse v4, PostgreSQL, ClickHouse, Redis, MinIO | Visualisation des traces | 3000 | — |

Dépendances : `frontend` attend `backend` sain, `backend` attend `mongodb` et `milvus`
sains. Ollama est joint par `host.docker.internal:11434`.

```
navigateur ──► frontend (nginx :8080) ──► /api/ ──► backend (:8000) ──► mongodb, milvus,
                                                                         Stockfish (local),
                                                                         Ollama (hôte),
                                                                         Lichess, YouTube
```

Le `Dockerfile` du front compile d'abord la bibliothèque `ngx-chess-board` (projet
`projects/`, résolue par `dist/ngx-chess-board` dans `tsconfig.json`) puis l'application en
configuration production ; nginx sert `index.html` pour toute route inconnue (application à
page unique) et met en cache un an les fichiers à empreinte (`*.js`, `*.css`).

## 2. Volumes persistants

| Volume | Contenu | Taille observée |
|---|---|---|
| `p13_mongodb_data` | MongoDB (corpus brut, cache vidéos, quota) | 231 Mo |
| `p13_milvus_data`, `p13_milvus_etcd`, `p13_milvus_minio` | collection `openings` (vecteurs, métadonnées) | 145 Mo + 11 Mo |
| `p13_hf_models` | modèle d'embedding Qwen3-Embedding-0.6B, téléchargé une fois | 1,2 Go |
| `p13_langfuse_*` (5) | données Langfuse, profil optionnel | — |

Contrôle : `docker volume ls`, puis `docker volume inspect p13_mongodb_data` (point de
montage), ou `docker system df -v` pour les tailles.

**Test de persistance** (`docker compose down` puis `docker compose up -d`) : voir § 5,
le résultat est consigné avec la date.

## 3. Installation fraîche

```bash
git clone <dépôt> && cd P13
cp .env.example .env            # renseigner LICHESS_TOKEN, YOUTUBE_API_KEY (facultatifs)
ollama pull qwen3:14b           # modèle de l'agent, sur la machine hôte
ollama pull phi4-mini           # résumés d'enrichissement (ingestion)
docker compose up -d --build    # frontend, backend, mongodb, milvus
docker compose exec backend uv run python -m scripts.ingest_corpus   # corpus → Milvus (≈ 1 min)
open http://localhost:8080
```

Sans jeton Lichess, l'Explorer répond quand même (limites de débit plus basses) ; sans clé
YouTube, la carte Vidéos indique que la recherche n'est pas configurée ; sans Ollama, les
cartes « Coups des maîtres », « Analyse profonde » et « Vidéos » fonctionnent, seul le
conseil rédigé et les questions libres échouent avec un message.

## 4. Positions de démo

| Scénario | Comment l'obtenir | Ce que l'on voit |
|---|---|---|
| **En théorie** | 1.e4 e5 2.Cf3 Cc6 3.Fc4 (ou FEN `r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3`) | « Italian Game · C50 », Fc5 25 k parties, 3 vidéos françaises, conseil avec évaluation rapide et extraits de l'encyclopédie |
| **Sicilienne** | 1.e4 c5 2.Cf3 d6 3.d4 | nom ECO, statistiques, vidéos sur la Sicilienne |
| **Hors théorie** | 1.a4 h5 2.Ta3 Th6 3.Tg3 | badge « Hors théorie », conseil fondé sur Stockfish (profondeur 18, 3 lignes), pas de vidéo |
| **Position invalide** | FEN `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0` (champ manquant) | message « FEN invalide » côté front, sans appel à l'API |
| **Analyse profonde** | n'importe quelle position, bouton « Analyse profonde » | suivi de la tâche de fond toutes les 2 s, résultat à la profondeur 28 |
| **Question libre** | « Quelles sont les idées de la Caro-Kann ? » | réponse rédigée à partir de l'encyclopédie |

## 5. Journal des vérifications

| Date | Vérification | Résultat |
|---|---|---|
| 22 oct. 2026 | Image frontend construite (bibliothèque puis application), conteneur `healthy`, `GET /` 200, `GET /api/v1/healthcheck` via nginx 200, `Cache-Control: immutable` sur les fichiers à empreinte | ✅ |
| 22 oct. 2026 | Application complète en conteneurs ouverte dans le navigateur (`http://localhost:8080`) : coups des maîtres chargés depuis Lichess via nginx → backend | ✅ |
| 22 oct. 2026 | Volumes déclarés et présents (`docker volume ls`) | ✅ 10 volumes `p13_*`, tailles ci-dessus |
| 22 oct. 2026 | **Persistance** : `docker compose down` (0 conteneur restant) puis `docker compose up -d`, backend et frontend sains après 15 s | ✅ avant / après identiques : MongoDB 12 articles, 11 entrées de cache vidéos, compteur de quota 2 211 ; Milvus répond à une recherche vectorielle avec 2 passages |
