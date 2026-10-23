# Processus de l'agent IA d'ouvertures (FFE)

Ce document fixe le fonctionnement de l'agent (Partie 1 — POC à coder) et le cadrage
de l'étude du serveur MCP de vision (Partie 2 — à concevoir uniquement).

---

## 1. Ce que l'agent fait pour le joueur

À chaque coup joué, l'agent répond à 4 questions. Chaque question correspond à un outil.

| Question du joueur | Outil | Ce qu'on renvoie |
|---|---|---|
| Où j'en suis ? | **FEN** (python-chess) | Position identifiée et validée |
| Quels sont les bons coups ? | **Lichess Opening Explorer** | Coups théoriques, % de victoires, nom de l'ouverture (code ECO) |
| Pourquoi cette ouverture ? | **Milvus (RAG Wikichess)** + parties de référence Lichess | Idées, plans, histoire, parties de maîtres |
| Montre-moi | **YouTube Data API** | 1 à 3 vidéos pertinentes |
| Je suis sorti de la théorie, et maintenant ? | **Stockfish** | Évaluation chiffrée et meilleurs coups |

---

## 2. Le processus à chaque coup

```
 Joueur joue un coup (Angular / ngx-chess-board)
                │
                ▼
 [1] Identifier la position → FEN (validée par python-chess)
                │
                ▼
 [2] Interroger Lichess Explorer avec la FEN
                │
        ┌───────┴────────┐
   EN THÉORIE        HORS THÉORIE
   (coups connus,    (0 coup connu, ou moins de N parties ;
    N parties ou plus) N ≈ 20, à calibrer)
        │                 │
        ▼                 ▼
 [3a] Meilleurs coups  [3b] Stockfish : éval + 3 meilleurs coups
      + nom ECO              (profondeur fixe, ex. 18)
        │                 │
        ▼                 │
 [4] RAG Milvus : contexte de l'ouverture (Wikichess)
        │                 │
        ▼                 │
 [5] YouTube : vidéos sur le nom de l'ouverture
        │                 │
        └───────┬─────────┘
                ▼
 [6] LLM : rédige une réponse pédagogique adaptée à un jeune joueur
                │
                ▼
 Réponse affichée : coups conseillés + explication + vidéos + éval
```

**Règle de décision :** une position est « hors des sentiers battus » quand Lichess ne
connaît aucun coup suivant, ou trop peu de parties. Dans ce cas, Stockfish devient la
source principale.

**Stockfish est appelé dans les deux branches** (critère d'évaluation : « Stockfish évalue
les positions théoriques ») :

- **en théorie**, il donne une évaluation rapide (profondeur réduite) en complément des coups Lichess ;
- **hors théorie**, il est la source principale : évaluation plus profonde et 3 meilleurs coups.

### Graphe LangGraph (implémenté : `backend/app/agent/graph.py`)

Deux entrées, selon que la requête porte une position (FEN) ou non :

```
START ─ FEN fournie ?
  ├─ oui ─► inspect ► theory (Lichess) ─ en théorie ?
  │                    ├─ oui ─► quick_eval (Stockfish, profondeur 12) ► context (RAG) ► videos ► answer
  │                    └─ non, ou Lichess en panne ─► engine (Stockfish, profondeur 18, 3 lignes) ► answer
  └─ non (question libre) ─► decide ⇄ tools ► answer
```

- **Parcours « position »** : déterministe, aucun choix laissé au modèle. C'est la logique
  de l'énoncé : Lichess d'abord ; si aucun coup théorique (ou si Lichess est indisponible),
  Stockfish devient la source principale. En théorie, Stockfish est aussi appelé en
  complément à profondeur réduite. Le LLM ne fait que **rédiger** le conseil à partir des
  résultats d'outils. Exposé par `POST /api/v1/agent/analyze`.
- **Parcours « question »** : le modèle décide des outils (ou flot fixe pour comparaison,
  voir `docs/outil-rag.md`, phase 5). Exposé par `POST /api/v1/agent/ask`.

---

## 2 bis. Mode « Jouer contre l'agent »

L'application propose deux modes :

| Mode | Qui joue | Rôle de l'agent |
|---|---|---|
| **Analyse** (section 2) | Le joueur joue les deux camps | Conseille sur la position |
| **Partie d'entraînement** | Le joueur contre l'agent | Joue l'adversaire **et** corrige les coups du joueur |

### Principe : séparer « jouer » et « expliquer »

- **Le choix du coup de l'agent** est fait par du code, **sans LLM** : c'est rapide,
  toujours légal et reproductible.
- **Le LLM sert uniquement à commenter** : correction du coup du joueur et bilan de fin de partie.

### Comment l'agent choisit son coup

```
 Position après le coup du joueur
                │
                ▼
 Lichess Explorer : la position est-elle en théorie ?
        │                         │
       OUI                       NON
        │                         │
        ▼                         ▼
 Tirage pondéré parmi les    Stockfish à niveau limité
 coups théoriques (selon     (option UCI « Skill Level » 0-20
 leur fréquence) → l'agent   ou « UCI_LimitStrength » + « UCI_Elo »)
 joue « comme les humains »  → adapté au niveau du jeune joueur
        │                         │
        └────────────┬────────────┘
                     ▼
          Coup joué sur l'échiquier
```

Le tirage pondéré fait varier les ouvertures d'une partie à l'autre. Le joueur
rencontre ainsi les lignes les plus jouées, et pas une seule ligne apprise par cœur.

**Option répertoire :** le joueur choisit l'ouverture qu'il veut travailler (ex. « Sicilienne
Najdorf avec les Noirs »). L'agent joue alors en priorité les coups qui restent dans cette ouverture.

### Comment l'agent corrige le joueur

Après chaque coup du joueur :

1. **Est-ce un coup théorique ?** Si Lichess le connaît, on affiche « Coup de théorie ✔ »
   avec le nom de l'ouverture.
2. **Sinon, on mesure la perte avec Stockfish** : évaluation du meilleur coup moins
   évaluation du coup joué, en centipions.

| Perte (centipions) | Verdict affiché |
|---|---|
| 0 à 50 | Bon coup |
| 50 à 100 | Imprécision |
| 100 à 300 | Erreur |
| Plus de 300 | Gaffe |

3. **En cas d'erreur, le LLM explique** pourquoi c'était faible et quel coup était
   meilleur. Il s'appuie sur le contexte RAG et peut proposer une vidéo.
4. **Option « reprendre le coup »** : le joueur peut annuler et réessayer. C'est utile
   pour l'apprentissage.

### Fin de partie : le bilan

Au bout de N coups (ex. 15 coups, la phase d'ouverture) ou à la fin de la partie, le LLM produit un bilan :

- à quel coup le joueur est sorti de la théorie ;
- ses erreurs principales et les meilleurs coups à leur place ;
- l'ouverture jouée, ses idées clés, et une vidéo pour réviser.

### Mise en pratique technique

**API (FastAPI) :**

| Route | Rôle |
|---|---|
| `POST /api/v1/games` | Crée une partie : couleur du joueur, niveau, ouverture visée (optionnelle) |
| `POST /api/v1/games/{id}/moves` | Reçoit le coup du joueur, renvoie la correction et le coup de l'agent |
| `POST /api/v1/games/{id}/undo` | Reprend le dernier coup |
| `GET /api/v1/games/{id}/summary` | Bilan de la partie |

**Données (MongoDB) :** une collection `games` contient l'id, la couleur, le niveau,
l'ouverture visée, la liste des coups (FEN, coup, verdict, évaluation) et le statut.

**LangGraph :** un second graphe `play_graph` réutilise les mêmes outils :

```
validate_move → grade_player_move → choose_agent_move → [explain_if_mistake] → save_state
```

Le nœud `explain_if_mistake` (LLM) n'est appelé que si le coup est une erreur. Cela garde
la partie fluide et limite le coût du LLM.

**Front (Angular) :**

- écran de démarrage : choix du mode, de la couleur, du niveau et de l'ouverture ;
- échiquier : il bloque le camp de l'agent et joue son coup avec une petite pause ;
- panneau latéral : verdict du coup, explication, bouton « reprendre », vidéo ;
- bilan affiché en fin de partie.

---

## 3. Étapes du projet (alignées sur l'énoncé)

Le backend FastAPI et MongoDB sont déjà en place (prérequis de l'étape 1).

| # | Étape | Résultat attendu (énoncé) | Planning |
|---|---|---|---|
| 1 | Services Lichess et Stockfish | `GET /api/v1/moves/{fen}` et `GET /api/v1/evaluate/{fen}` | S1 J1-2 |
| 2 | RAG Wikichess → Milvus | Script d'ingestion + endpoint `/vector-search` | S1 J3-4 |
| 3 | Agent LangGraph + YouTube + mode partie | Graphe complet, route agent, routes `/games` | S1 J5 → S2 J2 |
| 4 | Front Angular | Échiquier `ngx-chess-board` + panneau de recommandations | S2 J3-4 |
| 5 | Étude MCP vision (sans code) | Note de 8 à 10 pages (voir section 4) | En parallèle |
| 6 | Containerisation et démo | `docker-compose.yml` complet, README, démo prête | S2 J5 |

> L'énoncé de l'étape 3 (agent + YouTube) n'est pas dans les consignes fournies :
> le contenu ci-dessus vient du besoin global (section 1 et 2).

### Étape 1 : Services Lichess et Stockfish

**À livrer :**

- `GET /api/v1/moves/{fen}` : coups théoriques depuis le Lichess Opening Explorer
  (coup en SAN et UCI, nombre de parties, % victoires blancs / nulles / noirs, nom et code ECO).
- `GET /api/v1/evaluate/{fen}` : évaluation Stockfish en centipions (ou « mat en N »),
  meilleur coup, profondeur utilisée.

**Organisation du code :**

```
backend/app/
├── api/routes/moves.py        # route fine : valide, appelle le service, renvoie
├── api/routes/evaluate.py
├── services/fen.py            # validation FEN avec python-chess
├── services/lichess.py        # LichessService : client httpx, timeout, retry, cache
├── services/stockfish.py      # StockfishService : moteur UCI via python-chess
└── schemas/moves.py, evaluation.py   # modèles Pydantic des réponses
```

**Points techniques :**

- **FEN dans l'URL :** une FEN contient des `/` et des espaces. Il faut déclarer la route avec
  `{fen:path}` dans FastAPI, et le front doit encoder la FEN (`encodeURIComponent`).
- **Validation :** `chess.Board(fen)` puis `board.is_valid()`. Si la FEN est invalide,
  on renvoie une erreur **422** avec un message clair.
- **Timeouts Lichess :** timeout de 5 s environ. Si Lichess est en panne ou dépasse le délai,
  on renvoie **503** ou **504**, jamais une erreur 500 brute.
- **Limites de requêtes :** si Lichess répond **429**, on attend au moins 1 minute avant de
  rappeler. On met les réponses en cache dans MongoDB (clé = FEN) : une position d'ouverture
  ne change pas, ce qui évite la majorité des appels.
- **Jeton Lichess :** prévoir une variable `LICHESS_TOKEN` dans `.env`. L'Explorer peut
  exiger une authentification (à vérifier sur la documentation au moment du code).
- **Stockfish :** ajouté au conteneur backend (paquet `stockfish`) ; profondeur ou temps
  limité (ex. profondeur 18 ou 1 s) pour garder des réponses rapides.
- **Tests :** pytest avec appels Lichess simulés (mock), y compris timeout et 429.

### Étape 2 : RAG Wikichess → Milvus

**À livrer :**

- Un script `backend/scripts/ingest_wikichess.py` qui récupère, découpe, vectorise et charge les textes.
- Un endpoint `/api/v1/vector-search?query=...` qui renvoie les passages les plus pertinents
  (texte, titre de l'article, source, score).

**Processus du script :**

1. Récupérer 5 à 10 articles Wikichess sur les ouvertures populaires : Italienne, Espagnole,
   Sicilienne, Française, Caro-Kann, Gambit Dame, Indienne du Roi, Anglaise…
2. Nettoyer le texte (menus, références, HTML) et garder la source (URL) de chaque article.
3. Découper en morceaux d'environ 500 tokens avec un chevauchement de 50 à 100 tokens,
   en respectant les titres de section. Chaque morceau garde le nom de l'ouverture.
4. Vectoriser avec `sentence-transformers` et le modèle `Qwen/Qwen3-Embedding-0.6B`.
5. Charger dans une collection Milvus `openings` : `id`, `embedding`, `text`, `opening`,
   `section`, `source_url`. Index HNSW, distance cosinus.
6. Stocker aussi les textes bruts dans MongoDB (traçabilité, ré-indexation possible).

**Points techniques :**

- Dans Docker, l'API joint Milvus par le nom du service (`milvus:19530`), jamais `localhost`.
  Milvus standalone a besoin d'`etcd` et de `minio` dans le docker-compose.
- La qualité dépend du découpage : tester 5 questions types (« idées de la Sicilienne »,
  « pièges de l'Italienne »…) et ajuster la taille des morceaux.
- Le script est **idempotent** : relancé deux fois, il ne crée pas de doublons.
- LangGraph peut orchestrer recherche Milvus puis mise en forme de la réponse
  (réutilisé par l'agent à l'étape 3).

### Étape 3 : Agent LangGraph, YouTube et mode partie (agent ✅ `docs/outil-rag.md` phase 5 ; YouTube ✅ `docs/outil-videos.md` ; mode partie ⏳)

- Graphe d'analyse (section 2) et graphe de partie (section 2 bis).
- Service YouTube avec cache MongoDB : une recherche coûte 100 unités sur un quota
  de 10 000 par jour, soit environ 100 recherches par jour.
- Tests de bout en bout sur 3 positions : en théorie, sortie de théorie, position invalide.

### Étape 4 : Front Angular ✅ (réalisation détaillée dans `docs/frontend.md`)

**À livrer :** échiquier `ngx-chess-board`, panneau de recommandations (coups suggérés,
contexte, vidéos), communication avec l'API par des services Angular.

**Organisation :**

```
frontend/src/app/
├── services/api.service.ts        # HttpClient : moves, evaluate, vector-search, agent, games
├── services/game-state.service.ts # état de la partie (FEN courante, historique) — source unique
├── components/board/              # ngx-chess-board
├── components/advice-panel/       # coups suggérés, évaluation, contexte, vidéos
└── components/mode-selector/      # analyse / partie contre l'agent
```

**Points techniques :**

- **Chargement et erreurs :** chaque appel affiche un indicateur ; une erreur réseau affiche
  un message sans bloquer l'échiquier.
- **Synchronisation :** la FEN courante est stockée à un seul endroit (`game-state.service`).
  On ignore une réponse de l'agent si la position a changé entre-temps (`switchMap` en RxJS
  annule la requête précédente).
- **Moins d'appels :** un appel seulement après un coup terminé, et un cache local par FEN.

### Étape 6 : Containerisation et démo ✅ (réalisation dans `docs/deploiement.md`)

**`docker-compose.yml` complet :** `frontend`, `backend` (avec Stockfish), `mongodb`,
`milvus` (+ `etcd`, `minio`). Chaque service a un healthcheck et des `depends_on`.

**Volumes persistants à vérifier :**

1. Volumes déclarés : `mongodb_data`, `milvus_data`, `etcd_data`, `minio_data`.
2. Contrôle : `docker volume ls` puis `docker volume inspect <volume>`.
3. Test de persistance : `docker compose down` puis `docker compose up -d`,
   et vérifier que les données Mongo et la collection Milvus sont toujours là.

**Test depuis une installation fraîche :** clone du dépôt, `cp .env.example .env`,
`docker compose up -d --build`, ingestion Wikichess, puis parcours complet dans le navigateur.

**Positions de démo à préparer :**

| Scénario | Position | Ce que la démo montre |
|---|---|---|
| En théorie | Italienne après 1.e4 e5 2.Cf3 Cc6 3.Fc4 | Coups Lichess, contexte RAG, vidéos |
| Sicilienne | Après 1.e4 c5 2.Cf3 d6 3.d4 | Ouverture riche, nom ECO, statistiques |
| Hors théorie | Coups inhabituels (ex. 1.a4 h5 2.Ta3 Th6 3.Tg3), à vérifier dans l'Explorer | Bascule vers Stockfish |
| Position invalide | FEN mal formée | Message d'erreur propre |
| Mode partie | Partie de 10 coups contre l'agent | Verdicts, explication d'erreur, bilan |

---

## 4. Partie 2 : serveur MCP « vidéo par position » ✅ (note rédigée : `docs/note-mcp-vision.md`, PDF `docs/note-mcp-vision.pdf`)

**Le problème :** la recherche YouTube par texte renvoie par exemple une vidéo de
45 minutes sur la Sicilienne, alors que le joueur veut l'explication du coup en cours.

**La solution :** retrouver la seconde exacte de la vidéo où la position actuelle
apparaît à l'écran.

### A. Indexation (hors ligne, en lot)

1. Sélectionner les vidéos (chaînes partenaires ou listes validées).
2. Extraire les frames : 1 image toutes les 1 à 2 s, ou seulement aux changements de scène (ffmpeg).
3. Détecter s'il y a un échiquier sur la frame (petit modèle de détection, type YOLO).
4. Convertir l'échiquier en FEN avec un modèle de vision « board-to-FEN ».
5. Valider la position avec python-chess et rejeter les positions illégales.
6. Regrouper les frames consécutives qui montrent la même FEN en un segment `[début, fin]`.
7. Stocker `{fen_clé, video_id, t_début, t_fin, confiance}` en base, avec un index sur la FEN.

### B. Recherche (en ligne, via MCP)

L'agent LangGraph est client MCP et appelle les outils exposés :

| Outil MCP | Rôle |
|---|---|
| `find_video_by_fen(fen)` | Renvoie les segments correspondants, ex. `youtube.com/watch?v=XXX&t=754s` |
| `list_segments(video_id)` | Liste les positions indexées d'une vidéo |
| `index_video(url)` | Ajoute une vidéo au pipeline d'indexation (administration) |

Si aucune vidéo ne correspond, on revient à la recherche texte YouTube du POC.

**Point technique clé :** une image ne montre que le placement des pièces. Le trait,
les roques et la prise en passant n'y sont pas visibles. On indexe donc uniquement le
1er champ de la FEN. Il faut aussi gérer l'échiquier retourné (affiché côté Noirs).

### Livrable 1 : note bénéfices et limites

| Bénéfices | Limites |
|---|---|
| Vidéo calée sur le coup exact, ce qui fait gagner beaucoup de temps | **Droits** : télécharger des vidéos YouTube est interdit par leurs conditions d'utilisation. Il faut l'accord des créateurs ou du contenu FFE. C'est le point bloquant n°1 |
| Le catalogue devient un actif propre à la FFE | Erreurs de vision : flèches, surlignages, styles d'échiquier variés, échiquiers physiques filmés |
| MCP réutilisable par d'autres apps ou agents | Le contexte manque : le commentaire audio n'est pas analysé (amélioration possible avec une transcription Whisper) |
| Fonctionne aussi hors théorie si une vidéo a montré la position | Couverture limitée par la taille du catalogue |

### Livrable 2 : schéma d'architecture

Le schéma doit montrer :

- l'app Angular, le backend FastAPI et l'agent LangGraph (client MCP) ;
- le **serveur MCP Vision** et ses outils ;
- l'index de positions (MongoDB ou Postgres) ;
- le pipeline d'indexation : file de tâches, workers GPU, ffmpeg, détecteur d'échiquier, modèle board-to-FEN ;
- un stockage temporaire des vidéos, supprimées après traitement.

### Livrable 3 : faisabilité et coûts

Ordres de grandeur, à présenter comme des hypothèses :

- **Calcul :** 1 000 vidéos de 20 min, échantillonnées à 1 frame toutes les 2 s, donnent
  environ 600 000 frames. À environ 30 ms par frame sur GPU, cela fait environ 5 h de GPU,
  soit quelques euros à quelques dizaines d'euros en cloud.
- **Fonctionnement :** une petite VM pour le serveur MCP et la base, environ 20 à 50 €/mois.
  Les coûts du LLM dépendent du nombre de requêtes.
- **Mise en place (le vrai coût) :** fine-tuning ou validation du modèle board-to-FEN,
  constitution d'un jeu de test annoté et négociation des droits. Environ 20 à 40 jours-homme.
- **Conclusion attendue :** c'est techniquement faisable et peu coûteux à faire tourner.
  Le risque principal est juridique, et la précision de la vision reste à mesurer.

L'énoncé demande de séparer **coûts de build** (mise en place, en jours-homme et en
infrastructure ponctuelle) et **opex** (coûts mensuels : hébergement, stockage, GPU
pour les nouvelles vidéos, LLM, API YouTube). Les estimations doivent rester réalistes,
avec une hypothèse basse et une hypothèse haute.

### Plan de la note (8 à 10 pages)

| § | Contenu | Pages |
|---|---|---|
| 1 | Contexte : limite de la recherche YouTube par texte | 0,5 |
| 2 | Solution proposée et parcours utilisateur | 1 |
| 3 | État de l'art : détection d'échiquier (OpenCV : contours, lignes de Hough, homographie ; modèles : YOLO, projets open source type chesscog, LiveChess2FEN, modèles board-to-FEN) | 1,5 |
| 4 | Architecture technique MCP (schéma + rôle de chaque brique, serveur avec FastMCP) | 1,5 |
| 5 | Bénéfices attendus | 1 |
| 6 | Limites et risques techniques et business | 1,5 |
| 7 | Faisabilité et coûts : build + opex, hypothèses basse et haute | 1,5 |
| 8 | Alternatives (voir ci-dessous) | 0,5 |
| 9 | Étapes de développement et recommandation | 0,5 |

**Architecture modulaire :** chaque étape du pipeline (extraction de frames, détection,
board-to-FEN, indexation) est un module remplaçable. On peut changer de modèle de
vision sans toucher au serveur MCP ni à l'agent.

**Alternatives à proposer :**

1. **Transcription audio** : Whisper transcrit la vidéo, un LLM repère les coups annoncés
   (« Cavalier f3 ») et reconstruit les positions avec horodatage. Pas de vision, moins cher,
   mais dépend de la qualité du commentaire.
2. **Chapitres et métadonnées** : utiliser les chapitres YouTube et les descriptions (souvent
   un PGN ou une liste de coups), ou un partenariat avec des créateurs qui fournissent
   le PGN horodaté. Le plus simple et sans risque juridique, mais la couverture est limitée.

**Étapes de développement proposées :**

1. **Preuve de faisabilité** : 20 vidéos annotées à la main, mesure de la précision board-to-FEN.
2. **MVP** : 200 vidéos d'un partenaire, serveur MCP avec `find_video_by_fen`.
3. **Industrialisation** : file de tâches, indexation des nouvelles vidéos en continu, monitoring.
4. **Enrichissement** : ajout de la transcription audio (alternative 1) pour le contexte.

---

## 5. Décisions à valider

1. Seuil « hors théorie » : proposition à moins de 20 parties dans la base Lichess.
2. ~~Stockfish seulement hors théorie ?~~ **Tranché** : Stockfish est appelé dans les deux
   branches (exigé par la grille d'évaluation).
3. Format de la note MCP : Markdown converti en PDF, ou document Word.
4. Mode partie : niveaux de Stockfish proposés (ex. 3 niveaux : débutant, club, avancé).
5. Mode partie : longueur de la séance d'entraînement (proposition : 15 coups, puis bilan).

---

## 6. Grille de réussite : comment prouver chaque critère

À cocher avant la soutenance. Chaque critère a une **preuve à montrer** à l'évaluateur.

### Système LangGraph, FastAPI, Milvus, MongoDB

| ✓ | Critère | Preuve à montrer | Étape |
|---|---|---|---|
| ☐ | Graphe LangGraph bien structuré | Image du graphe générée par LangGraph (`get_graph().draw_mermaid()`) mise dans le README ; un nœud = une responsabilité | 3 |
| ☐ | L'agent traite une FEN et choisit la bonne source | Tests : position en théorie → branche Lichess ; hors théorie → branche Stockfish ; FEN invalide → erreur propre | 3 |
| ☐ | Bonnes pratiques Python | `ruff` (lint + format), annotations de types, docstrings, `pytest` qui passe ; idéalement lancés en CI (GitHub Actions) | Toutes |
| ☐ | Logique métier séparée de l'API | Routes fines dans `api/routes/`, logique dans `services/`, modèles dans `schemas/` ; les services sont testables sans FastAPI | 1 → 3 |
| ☐ | Données Wikichess prétraitées et découpées sans erreur | Logs du script d'ingestion : nb d'articles, nb de morceaux, 0 erreur ; script relançable sans doublon | 2 |
| ☐ | Embeddings avec un modèle approprié, stockés dans Milvus | Justifier le choix (`Qwen3-Embedding-0.6B` : multilingue, léger, bon score MTEB) ; compter les entités dans la collection | 2 |
| ☐ | Recherche vectorielle pertinente | Jeu de 10 questions de test avec l'ouverture attendue ; mesurer le % de bonnes réponses dans le top 3 | 2 |
| ☐ | Milvus connecté au workflow LangGraph | Nœud `rag_context` qui appelle le service Milvus ; visible dans le graphe et dans la réponse de l'agent | 3 |
| ☐ | Réponses de l'agent pertinentes | 5 positions de référence avec la réponse attendue, relues et validées ; garder les exemples pour la démo | 3 |
| ☐ | Lichess renvoie les coups théoriques | `GET /api/v1/moves/{fen}` testé sur l'Italienne et la Sicilienne | 1 |
| ☐ | Stockfish évalue les positions (y compris théoriques) | `GET /api/v1/evaluate/{fen}` + évaluation présente dans les deux branches de l'agent | 1, 3 |
| ☐ | YouTube renvoie des vidéos pertinentes | Requête construite à partir du nom ECO de l'ouverture (+ « explication », langue `fr`) ; filtre sur la durée (vidéos courtes ou moyennes) ; cache | 3 |
| ☐ | L'agent choisit des outils pertinents | Routeur conditionnel explicite + traces des appels (logs ou LangSmith) montrées en démo | 3 |
| ☐ | Timeouts et erreurs d'API gérés | Tests avec mocks : timeout, 429, 500 → réponses 503 / 504 propres ; l'agent continue sans l'outil en panne | 1, 3 |

### Docker Compose

| ✓ | Critère | Preuve à montrer | Étape |
|---|---|---|---|
| ☐ | Tous les services démarrent | `docker compose ps` : tous « healthy » | 6 |
| ☐ | Les services communiquent | Healthchecks backend → MongoDB, Milvus ; front → backend ; parcours complet dans le navigateur | 6 |
| ☐ | Volumes persistants configurés | `docker volume ls` / `inspect` ; données présentes après `down` puis `up` | 6 |
| ☐ | Variables d'environnement dans la config | `.env.example` complet ; aucun secret dans le code ni dans Git (clés YouTube, Lichess, LLM) | Toutes |
| ☐ | Application accessible depuis l'extérieur | Front servi par nginx sur un port publié ; backend écoute sur `0.0.0.0` ; CORS configuré ; test depuis un autre appareil du réseau (ou déploiement sur une VM) | 6 |

### Interface Angular

| ✓ | Critère | Preuve à montrer | Étape |
|---|---|---|---|
| ☐ | Échiquier intégré | `ngx-chess-board` jouable, coups illégaux refusés | 4 |
| ☐ | FEN synchronisées | La FEN affichée = celle envoyée à l'agent = celle de l'échiquier ; réponse ignorée si la position a changé | 4 |
| ☐ | Recommandations pertinentes affichées | Panneau : coups suggérés, évaluation, contexte, vidéos | 4 |
| ☐ | Chargement et erreurs gérés | Indicateur pendant l'analyse ; message clair si le backend est coupé (à montrer en démo) | 4 |
| ☐ | Bonne expérience utilisateur | Interface simple pour un jeune joueur, 2 modes clairs, responsive | 4 |

### Note MCP : bénéfices, limites, faisabilité

| ✓ | Critère | Où dans la note |
|---|---|---|
| ☐ | Bénéfices du système d'analyse vidéo identifiés | § 5 |
| ☐ | Limites techniques et business évaluées | § 6 |
| ☐ | Architecture cohérente et réalisable | § 3 et § 4 (schéma MCP) |
| ☐ | Estimations de coûts pertinentes (build + opex) | § 7, hypothèses basse et haute |
| ☐ | Alternatives et roadmap | § 8 et § 9 |

### Soutenance

| ✓ | Critère | Préparation |
|---|---|---|
| ☐ | Présenter les livrables et en démontrer la pertinence | Support de présentation court + script de démo avec les 5 positions préparées (étape 6) ; savoir justifier chaque choix technique (Milvus, modèle d'embedding, seuil hors théorie, MCP) |
