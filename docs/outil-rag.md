# Outil RAG : recherche dans la base de connaissances

Ce document décrit la mise en place, la mesure et l'évolution de l'outil RAG de l'agent
(`search_opening_knowledge`) : du RAG simple demandé par l'énoncé jusqu'à une
recherche **agentique** et **multi-sauts**, instrumentée et évaluée.

Il commence par le **plan de travail**. Chaque phase sera complétée ici au fur et à mesure,
avec ses choix, ses résultats chiffrés et ses limites.

| Statut | Signification |
|---|---|
| ⏳ | À faire |
| 🚧 | En cours |
| ✅ | Fait et mesuré |

---

## 1. Décisions de départ

| Sujet | Décision | Raison |
|---|---|---|
| Hébergement | **Tout en local** : services en conteneurs (Docker Compose), sauf Ollama | Exigence du projet : aucun service cloud |
| Base vectorielle | **Milvus 2.6.25** standalone (+ `etcd`, `minio`), configuration officielle | Branche 2.6 maintenue et éprouvée ; Milvus 3.0 est trop récent (juillet 2026) |
| Client Milvus | `pymilvus` 2.6 | Aligné sur la version du serveur |
| Modèle d'embedding | **`Qwen/Qwen3-Embedding-0.6B`** via `sentence-transformers` | Recommandation de l'énoncé ; multilingue, 1024 dimensions |
| PyTorch | Version **CPU** dans le conteneur Linux | La version par défaut embarque CUDA (~2 Go inutiles sans GPU) |
| LLM | Ollama **sur la machine hôte**, hors Docker ; **phi4-mini** comme modèle principal, modèle de l'agent choisi par la mesure en phase 5 | Ollama local profite du GPU du Mac, inaccessible depuis Docker. Le backend le joint via `host.docker.internal:11434` |
| Traces | **OpenTelemetry** avec un journal de décision local (référence) ; export **Langfuse auto-hébergé optionnel** (profil Compose `tracing`, arrêté par défaut) | OpenTelemetry est le standard d'instrumentation ; Langfuse (6 conteneurs, ClickHouse gourmand) n'est lancé que pour visualiser des sessions, le journal local suffit aux mesures |
| Corpus | **Wikipédia FR** (10 articles d'ouvertures), licence CC BY-SA | Wikichess.fr ne contient pas d'articles exploitables (application d'une seule page, 4 pages dans son plan de site). L'énoncé accepte « toutes sources pertinentes » |

---

## 2. Principes de la démarche

1. **Le jeu de référence est construit avant tout le reste.** Sans questions annotées,
   impossible de comparer les découpages ni de mesurer un rappel.
2. **Un RAG simple sert de point de départ.** Chaque amélioration (découpage,
   enrichissement, agent, multi-sauts) est ensuite **mesurée** par rapport à lui.
3. **Les passages attendus sont annotés comme des extraits de texte**, pas comme des
   numéros de chunks. Le même jeu de référence reste ainsi valable quel que soit le
   découpage : un chunk est pertinent s'il contient l'extrait attendu.
4. **Une variable à la fois** : chaque comparaison ne change qu'un paramètre
   (taille, recouvrement, méthode de découpage, enrichissement).

---

## 3. Plan de travail

### Phase 0 : socle (infrastructure et corpus) ✅

| Tâche | Détail |
|---|---|
| Conteneurs | Milvus 2.6 (+ `etcd`, `minio`), backend. Ollama reste sur la machine hôte. Langfuse ajouté en phase 5. Volumes persistants, healthchecks, cache du modèle d'embedding dans un volume |
| Vérifier phi4-mini | Test de l'**appel d'outils** et de la vitesse. En cas d'échec : alternative locale proposée avant de continuer |
| Corpus | 10 articles Wikipédia FR (Italienne, Espagnole, Sicilienne, Française, Caro-Kann, Gambit dame, Est-indienne, Anglaise, Scandinave, Pirc…), récupérés en **3 formats** : texte, Markdown (titres `##`), HTML. Figés dans `backend/data/corpus/` |
| **Métadonnées filtrables** | `date` (date de révision), `version` (numéro de révision Wikipédia), `service_emetteur` (`wikipedia-fr`, `ffe`…), `confidentialite` (`public` / `interne`), `type_document`, `ouverture` |

#### Résultats de la phase 0

**Infrastructure** (`docker-compose.yml`) :

| Service | Image | Rôle |
|---|---|---|
| `milvus` | `milvusdb/milvus:v2.6.25` | Base vectorielle, port 19530 |
| `etcd` | `quay.io/coreos/etcd:v3.5.25` | Métadonnées de Milvus |
| `minio` | `milvusdb/minio:RELEASE.2024-12-18T13-15-44Z` | Stockage des segments de Milvus |
| `backend` | Image du projet (2,67 Go, dont PyTorch **CPU** 652 Mo) | API ; joint Milvus par `milvus:19530` et Ollama par `host.docker.internal:11434` |
| `mongodb` | `mongo:8.0` | Textes bruts et caches |

Volumes persistants ajoutés : `milvus_etcd`, `milvus_minio`, `milvus_data`, `hf_models`
(modèle d'embedding téléchargé une seule fois). Vérifié depuis le conteneur backend :
Milvus 2.6.25 répond, et les modèles d'Ollama de l'hôte sont visibles.

**Test du choix d'outil** (5 cas : 3 questions qui exigent un outil, 2 qui n'en exigent
pas ; Ollama local, température 0). Deux modes ont été comparés :
- **appel d'outils natif** : les outils sont passés à l'API `tools` d'Ollama ;
- **sortie structurée** : la génération est contrainte par un schéma JSON
  `{"tool": <nom> | null, "arguments": {…}, "reason": "…"}`.

| Modèle | Appel natif | Sortie structurée |
|---|---|---|
| phi4-mini | 2/5 (≈ 1 s) | **4/5** (1,4 s) |
| qwen3:1.7b | **5/5** (0,6 s) | 3/5 (1,1 s) |
| qwen3:14b | **5/5** (2,9 s) | non testé |

Constats :
- **phi4-mini choisit le bon outil mais l'écrit dans un format non standard** :
  `function("search_opening_knowledge", {…})` au milieu d'une phrase en français, ou la
  définition de l'outil recopiée en anglais. Le gabarit d'Ollama est pourtant conforme au
  format officiel de Microsoft (`<|tool|>…<|/tool|>`). Avec le schéma JSON imposé, le
  format n'est plus un problème : 4/5.
- **Le meilleur mode dépend du modèle** : qwen3 réussit mieux en appel natif ; en sortie
  structurée, il appelle la recherche même pour une question hors sujet.
- En appel natif, phi4-mini **invente** parfois le résultat de l'outil (une « sortie »
  Stockfish fictive) : à surveiller dans les réponses finales.

**Décision** : phi4-mini reste le modèle principal (génération de la phase 4). Le modèle
et le mode de l'agent seront **choisis par la mesure en phase 5**, sur Tâches-30 :
phi4-mini en sortie structurée, qwen3:1.7b et qwen3:14b en appel natif. Avec 5 cas,
un seul écart vaut 20 points : l'échantillon est trop petit pour trancher.

**Corpus** (`backend/data/corpus/`, 12 articles, 2,3 Mo, récupéré par
`scripts/fetch_corpus.py`) :

| Article | Ouverture (nom Lichess) | Texte | Révision |
|---|---|---|---|
| Partie italienne | Italian Game | 1 537 car. | 232955183 |
| Défense des deux cavaliers | Italian Game: Two Knights Defense | 12 132 car. | 240416129 |
| Gambit Evans | Italian Game: Evans Gambit | 6 538 car. | 239433390 |
| Partie espagnole | Ruy Lopez | 13 254 car. | 240543808 |
| Défense sicilienne | Sicilian Defense | 13 279 car. | 239611960 |
| Défense française | French Defense | 17 312 car. | 240051570 |
| Défense Caro-Kann | Caro-Kann Defense | 17 331 car. | 237180535 |
| Gambit dame | Queen's Gambit | 3 644 car. | 239865569 |
| Défense slave | Queen's Gambit Declined: Slav Defense | 12 161 car. | 232187085 |
| Défense est-indienne | King's Indian Defense | 9 436 car. | 238358295 |
| Ouverture anglaise | English Opening | 10 404 car. | 229966041 |
| Défense scandinave | Scandinavian Defense | 7 083 car. | 238972225 |

Choix et constats :
- **« Partie italienne » est trop court** (1 537 caractères) : il est complété par ses
  deux grandes branches (Deux cavaliers, Gambit Evans). Cela fournit aussi des chaînes
  naturelles pour les questions multi-sauts.
- **Nom d'ouverture Lichess** stocké avec chaque article : le filtre `opening` de l'outil
  de recherche utilise le même nom que celui renvoyé par `get_opening_theory`.
- **Texte et HTML viennent de la même révision** (`oldid`) : les trois formats sont
  cohérents entre eux.
- **Politique Wikimedia** : l'API exige un User-Agent avec un moyen de contact (sinon
  HTTP 403). Il est configurable (`WIKIPEDIA_CONTACT`) ; une pause de 1 s sépare les requêtes.
- **À traiter à l'ingestion (phase 2)** : deux notations des coups coexistent (« Cf3 » et
  « ♘f3 ») et doivent être normalisées en lettres françaises ; les sections « Notes et
  références » et « Bibliographie » sont à exclure de l'index.

**Tests** : 17 tests du module de corpus (API Wikipédia simulée avec respx) ; 219 tests
au total, couverture de branches 99 %.

### Phase 1 : jeux de référence ✅

| Jeu | Contenu | Sert à |
|---|---|---|
| **Réf-30** (« jeu M2 ») | 30 questions + extraits attendus annotés | recall@5 des découpages ; rappel et précision de l'agent |
| **Décision-50** | 50 questions, dont **20 hors corpus** | Taux d'appels inutiles |
| **Tâches-30** | 30 tâches avec l'outil attendu | Taux d'appels d'outil corrects |
| **Multi-10** | 10 questions multi-sauts avec leurs sous-questions attendues | Multi-sauts vs RAG en un seul passage |

#### Résultats de la phase 1

Les 4 jeux sont dans `backend/data/eval/` (JSON Lines). Le format est défini dans
`app/rag/eval/datasets.py`, et chaque jeu est construit par un script reproductible.

**Étape 1.1 : format et normalisation commune.** `app/rag/text.py` normalise le texte
pour l'ingestion **et** l'évaluation : figurines Unicode → lettres françaises
(« 4...♝c5 5.♘c3 » → « 4...Fc5 5.Cc3 »), espaces simples (y compris insécables).
Un chunk est jugé pertinent s'il **contient** l'extrait attendu, comparaison faite sur
une seule ligne, sans tenir compte de la casse (`contains_excerpt`) : le jugement ne
dépend donc pas du découpage.

**Étape 1.2 : passages candidats** (`app/rag/eval/passages.py`). Chaque article est
découpé en paragraphes avec leur chemin de section ; les sections « Notes et
références », « Bibliographie », « Liens externes »… sont écartées, ainsi que les
paragraphes de moins de 200 caractères. Résultat : 134 passages.
Un filtre « texte rédigé » écarte ensuite les listes de coups et les parties complètes :
si plus de 25 % des mots sont des coups (« 4.Cf3 », « exd4 », « O-O »), le passage est
écarté. Exemple : « 1.e4 c5 2.Cf3 e6 3.d4… » a un ratio de 1,00 ; une phrase rédigée,
environ 0,15. Il reste **78 passages rédigés**.

**Étape 1.3 : génération avec phi4-mini** (`app/rag/eval/generate.py`,
`scripts/generate_ref_candidates.py`). 3 passages par article, répartis sur tout
l'article (1 seul pour « Partie italienne »), soit 34 passages. Pour chacun, phi4-mini
produit en **sortie structurée** une question, l'extrait qui contient la réponse et la
réponse. Contrôle automatique immédiat : l'extrait doit exister **mot pour mot** dans le
passage ; sinon, seconde tentative avec le motif du refus.
Résultat : **34 candidats en 63 s** ; 29 validés par le contrôle, 5 rejetés (extrait
trop court ou paraphrasé).

**Étape 1.4 : relecture** (`scripts/build_ref30.py`, décisions tracées dans
`data/eval/ref30_review.jsonl`). Le contrôle automatique garantit que l'extrait existe,
pas que la question est bonne. Chaque candidat a été relu :

| Verdict | Nombre | Exemples de défauts corrigés |
|---|---|---|
| Validée (au plus une retouche de forme) | 6 | — |
| Corrigée | 22 | Question sans rapport avec l'extrait (6) ; question qui donne la réponse (5) ; détail inventé (2) ; réponse fausse (« Fa5 » au lieu de « Ca5 ») ; joueurs intervertis (Karpov / Kasparov) |
| Récupérée après rejet automatique | 2 | Extrait paraphrasé ; espace supprimée avant « ; » |
| Rejetée | 4 | 3 listes de coups ; 1 hors périmètre (série Netflix) |

**Leçon** : phi4-mini accélère le premier jet (choix du passage, ébauche), mais
**22 candidats sur 34 avaient un défaut** : une relecture humaine est indispensable.
Chaque extrait final est revérifié dans l'article complet.

**Réf-30** : 30 questions, les 12 articles couverts (1 à 3 questions chacun).

**Étape 1.5 : Décision-50** (`scripts/build_decision50.py`).

| Catégorie | Nombre | Recherche | Exemple |
|---|---|---|---|
| `dans_corpus` | 30 | Utile | Les 30 questions de Réf-30 |
| `echecs_hors_corpus` | 8 | Inutile, **pas toujours évitable** | « Quelles sont les idées du gambit letton ? » |
| `autre_outil` | 6 | Inutile, **évitable** | « Qui est mieux dans cette position : <FEN> ? » (→ `evaluate_position`) |
| `hors_sujet` | 6 | Inutile, **évitable** | « Quelle est la capitale de l'Australie ? » |

L'absence des 8 sujets hors corpus est **vérifiée par mot-clé** au moment de la
construction. Cette vérification a écarté plusieurs premières idées (Alekhine, Fischer,
nimzo-indienne, Grünfeld), qui apparaissent en passant dans le corpus.

**Étape 1.6 : Tâches-30** (`scripts/build_tasks30.py`). 30 demandes couvrant les
7 outils et le cas « aucun outil » : `inspect_position` 4, `get_opening_theory` 5,
`evaluate_position` 5, `search_opening_knowledge` 6, `find_opening_videos` 4,
`start_deep_analysis` 2, `get_deep_analysis` 1, aucun outil 3. Les arguments imposés
par la demande sont attendus (`lines=3` pour « les 3 meilleurs coups »,
`database=lichess` pour « les joueurs en ligne », `include_games=true`, `depth=35`).
Les 18 FEN sont **calculées** par python-chess à partir de suites de coups.

**Étape 1.7 : Multi-10** (`scripts/build_multi10.py`). 10 questions à 2 sauts ; la
sous-question du saut 2 utilise la réponse du saut 1. Exemple :
« Quel métier exerçait l'homme qui a donné son nom au gambit de la partie italienne joué
par 4.b4 ? » → saut 1 : 4.b4 = gambit Evans → saut 2 : William Davies Evans, capitaine
de navire gallois. **3 chaînes traversent deux articles** (française → sicilienne,
Caro-Kann → anglaise, slave → gambit dame) ; 7 relient deux sections d'un même article.

**Étape 1.8 : validation** (`app/rag/eval/validate.py`, `scripts/validate_eval.py`).
Contrôles : effectifs, identifiants uniques, présence de chaque extrait dans son article,
répartition de Décision-50, outils existants, FEN valides, dépendances entre sauts.
Le validateur tourne aussi dans pytest : un jeu incohérent fait échouer la suite.

**Tests** : 29 tests pour la phase 1 (normalisation, passages, génération avec Ollama
simulé par respx, validation et détection des incohérences). 248 tests au total,
couverture de branches 99 %.

**Reconstruire les jeux :**

```bash
cd backend
uv run python -m scripts.generate_ref_candidates   # phi4-mini, ~1 min
uv run python -m scripts.build_ref30
uv run python -m scripts.build_decision50
uv run python -m scripts.build_tasks30
uv run python -m scripts.build_multi10
uv run python -m scripts.validate_eval
```

**Limites :**
- **Un seul relecteur** pour Réf-30 : une relecture croisée par une seconde personne
  renforcerait le jeu.
- **Un seul extrait attendu par question** : si la même information figure à deux
  endroits du corpus, une recherche qui trouve l'autre passage sera comptée comme un échec.
- **Décision-50 réutilise les questions de Réf-30** pour sa partie « dans le corpus » :
  les deux mesures ne sont pas indépendantes.
- **Multi-10 a surtout des chaînes à l'intérieur d'un même article** (7 sur 10) : ces
  questions peuvent parfois être résolues en une seule recherche si les deux faits sont
  dans le même chunk. C'est justement ce que la phase 6 permettra de mesurer.
- **« Partie italienne » n'a qu'une question** (article de 1 537 caractères) ; la famille
  italienne est couverte par la Défense des deux cavaliers et le Gambit Evans.

### Phase 2 : RAG simple, la référence (livrables de l'énoncé) ✅

| Tâche | Détail |
|---|---|
| Script d'ingestion | `scripts/ingest_corpus.py` : nettoyage, découpage récursif 512 tokens, embeddings, chargement dans Milvus (identifiants stables : relançable sans doublons) et textes bruts dans MongoDB. Exécuté dans un conteneur |
| Outil | `search_opening_knowledge` branché sur Milvus et le vrai modèle (il répond aujourd'hui `not_configured`) |
| Endpoint | `GET /api/v1/vector-search?query=…&opening=…&limit=…` |
| Tests | Sans réseau (découpage, adaptateur Milvus, endpoint, ingestion) + test d'intégration avec le vrai Milvus |
| Mesure de référence | recall@5 sur Réf-30 |

#### Résultats de la phase 2

**Étape 2.1 : découpage de référence** (`app/rag/chunking.py`).
- **Récursif, 512 tokens, recouvrement 51 tokens (10 %)**, tokens comptés avec le
  tokenizer du modèle d'embedding : la limite porte sur ce que le modèle lit vraiment.
- **Séparateurs adaptés au français**, du plus fort au plus faible : titre Markdown →
  paragraphe → ligne → fin de phrase (`. ? ! …`) → `; :` → `,` → espace. Ce sont des
  expressions régulières qui ne coupent que l'espace (`(?<=[.!?…])\s+`) : avec un
  séparateur simple comme `". "`, le point se retrouverait au début du chunk suivant.
  Un titre ouvre toujours le chunk qui le suit.
- **Sections exclues** : « Notes et références », « Bibliographie », « Liens externes »…
- **Fusion des petits chunks** (moins de 50 tokens) avec leur voisin : le premier
  découpage produisait 10 chunks de moins de 60 tokens, dont un titre seul
  (« # Défense française », 5 tokens) et des phrases d'introduction orphelines
  (« Au choix des Blancs, on peut trouver : »). Après fusion : **148 chunks**, de 57 à
  552 tokens (moyenne 341, médiane 387). 3 chunks dépassent légèrement 512 tokens à
  cause des fusions, sans conséquence (le modèle accepte 32 768 tokens).
- Chaque chunk garde son **fil d'Ariane** (`section`) et les métadonnées de l'article.

**Étape 2.2 : embedder** (`app/rag/embedding.py`). `Qwen3-Embedding-0.6B`,
1 024 dimensions, vecteurs normalisés. Modèle **asymétrique** : consigne de recherche
pour les questions (`prompt_name="query"`), aucune pour les documents.

**Étape 2.3 : adaptateur Milvus** (`app/rag/milvus_store.py`). Collection `openings`,
index **HNSW** (M = 16, efConstruction = 200), distance **cosinus**. Les métadonnées
filtrables sont des champs scalaires : `date`, `version`, `service_emetteur`,
`confidentialite`, `type_document`, `opening`, `opening_family`, `strategy`. Identifiants
stables (`recursive:defense-slave:007`). Erreurs traduites dans la taxonomie : Milvus
injoignable → `vector_store_error` / `network_error` (réessayées) ; collection absente →
`not_indexed`, avec le conseil de lancer l'ingestion.

**Étape 2.4 : ingestion** (`app/rag/ingest.py`, `scripts/ingest_corpus.py`).
**148 chunks, 50 490 tokens, en ≈ 30 s** (Mac, GPU). Textes bruts dans MongoDB
(collection `corpus_articles`).
Un piège rencontré : après une 2ᵉ ingestion, Milvus annonçait **296** chunks. La
statistique `row_count` compte aussi les lignes supprimées pas encore compactées (un
`upsert` = suppression + insertion). Le comptage utilise donc une requête `count(*)`.
Vérifié : après deux ingestions, **148 chunks et 148 identifiants uniques**.

**Étapes 2.5 et 2.6 : outil et endpoint.** `search_opening_knowledge` est branché sur
Milvus et le vrai modèle ; `GET /api/v1/vector-search?query=…&opening=…&limit=…` passe
par le registre (validation, timeout, tentatives, disjoncteur).

**Le démarrage à froid**, découvert au premier essai réel : les 4 premières recherches
ont échoué en `timeout`. Le chargement du modèle (**8,4 s**) dépassait le timeout de
5 s de l'outil, et les appels suivants attendaient le verrou. Une fois le modèle chargé,
une recherche prend **50 à 100 ms** (embedding + Milvus). Corrections :
1. **Préchauffage** : l'API lance le chargement en arrière-plan dès son démarrage, sans
   retarder sa disponibilité.
2. **Réponse immédiate** pendant le chargement : `model_loading`, avec un délai de nouvel
   essai, au lieu d'attendre puis d'échouer. L'agent peut continuer avec les autres outils.

**Étape 2.7 : tests.** 42 tests ajoutés : découpage, embedder (faux modèle, y compris
`model_loading`), adaptateur Milvus (faux client : filtres, schéma, suppression avant
réinsertion, erreurs), ingestion, endpoint, métriques d'évaluation (recall@k, MRR,
precision@k, calibration). **290 tests, couverture de branches 99 %.**
Test d'intégration sur le vrai Milvus et le vrai modèle (ingestion, relance sans doublon,
bon article en tête, filtre) : **réussi** (`uv run pytest -m integration`).

**Étape 2.8 : mesure de référence** (`app/rag/eval/retrieval.py`,
`scripts/eval_retrieval.py`, résultats dans `data/eval/results/phase2-reference.json`).

| Variante | recall@1 | recall@3 | **recall@5** | MRR | precision@5 |
|---|---|---|---|---|---|
| Sans filtre | 0,50 | 0,70 | **0,77** | 0,60 | 0,15 |
| Avec filtre d'ouverture | 0,50 | 0,70 | **0,80** | 0,62 | 0,16 |

- **Contrôle de la mesure** : les 30 extraits attendus sont tous présents dans au moins
  un chunk de l'index. Une question manquée est donc un défaut de classement, pas un
  biais de l'évaluation.
- **7 questions manquées**, au même profil : leur réponse est dans une **liste de coups
  ou de variantes** (« 3...dxe4 (variante Rubinstein) », « Elle s'obtient après les coups
  5. a4 Ff5 »). Ces chunks contiennent peu de texte rédigé, et le modèle les rapproche mal
  d'une question en français. C'est la cible des phases 3 et 4 (fil d'Ariane en tête de
  chunk, questions hypothétiques).
- La precision@5 est plafonnée à 0,20 (un seul extrait attendu par question).

**Calibration du seuil de pertinence** (meilleur score de chaque question) :

| Population | Meilleurs scores |
|---|---|
| Questions du corpus (30) | 0,51 à 0,84 (médiane 0,74) |
| Hors sujet (6) | 0,21 à 0,37 |
| Échecs hors corpus (8) | 0,49 à 0,64 |
| Relevant d'un autre outil (6) | 0,64 à 0,73 |

| Seuil | Questions du corpus gardées | Questions hors corpus écartées |
|---|---|---|
| 0,40 | 100 % | 30 % |
| **0,50 (retenu)** | **100 %** | **35 %** (dont tous les hors-sujet) |
| 0,55 | 97 % | 50 % |
| 0,60 | 93 % | 65 % |
| 0,70 | 67 % | 85 % |

**Étape 2.9 : exécution dans Docker** ✅.
- **Image** reconstruite (2,67 Go). Un premier échec de `uv sync` était passager
  (téléchargement interrompu).
- **Ingestion dans le conteneur : fonctionnelle mais trop lente.** Modèle téléchargé une
  fois dans le volume `hf_models` (1,1 Go), embeddings calculés **sur CPU** (7 cœurs à
  100 %), Docker n'ayant pas accès au GPU du Mac : environ **9 minutes par article**
  (4 articles en 37 min), soit environ 1 h 30 pour le corpus, contre **31 s** sur l'hôte.
  Elle a été arrêtée une fois ce constat établi.
- **Décision** : l'ingestion, opération ponctuelle, se lance **depuis l'hôte** vers le Milvus
  conteneurisé (`uv run python -m scripts.ingest_corpus --recreate`) ; les données
  persistent dans le volume `milvus_data`. La commande dans le conteneur reste disponible
  pour une machine Linux ou une exécution de nuit.
- **Requêtes depuis le conteneur backend, mesurées depuis l'hôte** (`curl`) :
  pendant le préchauffage (≈ 12 s), `/vector-search` répond immédiatement
  `model_loading` ; ensuite, **0,5 à 0,8 s par recherche** de bout en bout (encodage
  d'une question sur CPU + Milvus), bien sous le timeout de 5 s.
- **Cohérence hôte / conteneur** : les scores sont identiques (0,7896 pour « Qui a donné
  son nom au gambit Evans ? »), les vecteurs calculés sur l'hôte et dans le conteneur sont
  compatibles.

**Décision : seuil 0,50.** Il garde toutes les bonnes réponses et écarte toutes les
questions hors sujet. Les questions d'échecs hors corpus ou relevant d'un autre outil ont
des scores aussi élevés que les vraies réponses (même vocabulaire) : **aucun seuil ne les
sépare** sans perdre de bonnes réponses. Les éviter relève de la **décision de l'agent**
(phase 5), pas du seuil.

### Phase 3 : comparer les découpages ✅

| Tâche | Détail |
|---|---|
| Familles de découpage | Fixe, récursif, structurel, sémantique, hiérarchique : principe, forces, limites |
| Récursif adapté au français | `RecursiveCharacterTextSplitter`, séparateurs : sections, paragraphes, phrases (`. `, `? `, `! `, `… `), `; `, `, ` |
| Structurel | `MarkdownHeaderTextSplitter` (titres) et découpage par sections HTML |
| Sémantique | `SemanticChunker` (LangChain), seuil au **95ᵉ percentile** de rupture de similarité entre phrases voisines |
| Hiérarchique | Petits chunks pour la recherche, rattachés à leur section parente pour le contexte |
| **Tailles** | 256 / 512 / 1024 tokens → **recall@5** sur Réf-30 |
| **Recouvrement** | 0 %, 10 %, 25 % → recall@5 et **redondance induite** (part de tokens dupliqués dans l'index) |
| Sémantique vs récursif | Nombre de chunks, longueur (moyenne et dispersion), **cohérence** (similarité moyenne entre phrases d'un même chunk) |
| **Late chunking** | Explication : encoder le document entier, puis moyenner les vecteurs de tokens par segment. Prototype si le modèle s'y prête (limite à documenter : Qwen3 est un modèle causal, peu adapté) |

#### Résultats de la phase 3

##### Les familles de découpage

| Famille | Principe | Forces | Limites |
|---|---|---|---|
| **Fixe** | Fenêtres de N tokens, sans regarder le texte | Tailles régulières, simple, prévisible | Coupe au milieu des phrases : il faut un recouvrement |
| **Récursif** | Essaie les séparateurs du plus fort au plus faible (section → paragraphe → phrase → mot) jusqu'à tenir dans la taille | Respecte la structure tant que possible ; standard | Tailles inégales ; le recouvrement ne joue qu'à l'intérieur d'un long paragraphe |
| **Structurel** | Une unité du document = un chunk (section Markdown, section HTML) | Chunks qui ont un sens ; coupe aux vrais changements de sujet | Tailles très inégales (de 17 à 4 405 tokens ici) |
| **Sémantique** | Encode chaque phrase et coupe là où la distance entre deux phrases voisines dépasse un seuil (ici le 95ᵉ percentile des distances de l'article) | Coupe là où le sens change, sans dépendre de la mise en forme | Coûteux (un embedding par phrase) ; tailles imprévisibles |
| **Hiérarchique** | Petits chunks (enfants) pour **chercher**, unité plus large (parent) **renvoyée** à l'agent | Recherche précise et contexte suffisant | Deux niveaux à gérer ; le parent doit être plafonné |

##### Méthode

`app/rag/eval/strategies.py` (stratégies), `app/rag/eval/bench.py` (banc),
`scripts/compare_chunking.py` (expériences), résultats dans
`data/eval/results/phase3-chunking.json`.

- **Même procédure pour toutes les stratégies** : découpage des 12 articles, encodage
  (cache disque : un texte n'est jamais encodé deux fois), recherche **exacte en mémoire**.
  On compare des découpages, pas des index : avec quelques centaines de chunks, le calcul
  exact est instantané et évite l'approximation de HNSW.
- **50 requêtes** : Réf-30 + les 20 sous-questions de Multi-10. Avec 30 seulement, une
  question vaut 3,3 points.
- **Mesures** : recall@5 et MRR ; **atteignables** (part des requêtes dont l'extrait est
  entier dans au moins un chunk : le plafond du recall) ; **tokens renvoyés** pour un top-5 ;
  **recall à budget égal** (meilleurs chunks pris dans l'ordre jusqu'à 2 000 tokens) ;
  **redondance** (part des tokens indexés en double à cause du recouvrement).
- **Significativité** : test de McNemar apparié contre la référence (récursif 512).
- Les dépendances `langchain-experimental` (SemanticChunker) et `beautifulsoup4` sont
  installées en **dépendances de développement** : elles ne servent qu'aux expériences.

##### A. Familles, à taille comparable

| Stratégie | Chunks | Tokens moy. (min-max) | Atteignables | recall@5 (50) | MRR | Tokens top-5 | recall à 2 000 tokens | vs référence |
|---|---|---|---|---|---|---|---|---|
| Fixe 512, recouvrement 51 | 112 | 488 (207-512) | 100 % | 0,80 | 0,68 | 2 469 | 0,76 | +4 / −3 |
| **Récursif 512 / 51 (référence)** | 148 | 341 (57-552) | 100 % | 0,78 | 0,65 | 1 788 | 0,78 | — |
| Titres Markdown (sections) | 110 | 451 (17-4 405) | 100 % | 0,80 | 0,66 | 2 169 | 0,80 | +2 / −1 |
| Sections HTML (brut) | 114 | 484 (7-4 446) | **28 %** | 0,16 | 0,13 | 2 699 | 0,16 | +2 / −33 |
| Sections HTML (balises de liens retirées) | 114 | 472 (10-4 392) | 100 % | 0,76 | 0,66 | 2 311 | 0,76 | +3 / −4 |
| Sémantique (percentile 95) | 180 | 276 (3-3 174) | 96 % | 0,68 | 0,61 | 2 179 | 0,68 | +1 / −6 |
| Hiérarchique (enfants 256 → section) | 323 | 158 (4-255) | 100 % | **0,86** | **0,70** | 2 593 | **0,84** | +4 / −0 |
| **Hiérarchique (enfants 256 → parent ≤ 1 024)** | 325 | 157 (4-256) | 100 % | **0,86** | **0,70** | **1 945** | **0,84** | **+4 / −0** |

« vs référence » : requêtes gagnées / perdues par rapport au récursif 512. Seul l'écart
du HTML brut est significatif (p < 0,01) ; les autres ont p ≥ 0,12.

**Le cas du HTML**, analysé en deux temps :
1. Dans le HTML de Wikipédia, chaque titre est enveloppé dans `<div class="mw-heading">`.
   `HTMLHeaderTextSplitter` ne rattache alors pas le texte suivant à son titre (métadonnées
   vides) : les sections de références n'étaient pas exclues. Corrigé par un prétraitement
   (`unwrap_wikipedia_headings`) ; effet faible sur le recall.
2. Le vrai problème : le découpeur **déplace le texte des liens à la fin du bloc**.
   « Le <a>gambit</a> porte le nom… » devient « Le porte le nom… \n gambit ». Les phrases
   sont cassées : seuls **28 %** des extraits attendus restent entiers. Une fois les balises
   de mise en forme retirées (`flatten_inline`), on passe à **100 %** d'extraits atteignables
   et **0,76** de recall, au niveau du découpage Markdown (mêmes sections).
   **Leçon** : partir du HTML exige un nettoyage spécifique à la source ; le Markdown issu de
   l'API texte est plus simple et aussi bon.

##### B. Taille des chunks (récursif, recouvrement 10 %)

| Taille | Chunks | Tokens moy. | recall@5 (50) | MRR | Tokens top-5 | **recall à 2 000 tokens** |
|---|---|---|---|---|---|---|
| 256 | 288 | 176 | 0,80 | 0,64 | **861** | **0,82** |
| 512 | 148 | 341 | 0,78 | 0,65 | 1 788 | 0,78 |
| 1 024 | 71 | 710 | **0,86** | **0,70** | 3 774 | 0,76 |

Le 1 024 obtient le meilleur recall@5 parce que chaque résultat contient **plus de texte** :
il renvoie 4,4 fois plus de tokens que le 256. **À quantité de texte égale, il est dernier**
(0,76) et le 256 premier (0,82). Le recall@5 seul favorise les gros chunks : il faut le lire
avec le coût en tokens.

##### C. Recouvrement et redondance

| Recouvrement | Fixe 512 : redondance | Fixe : atteignables | Fixe : recall@5 | Récursif 512 : redondance | Récursif : recall@5 |
|---|---|---|---|---|---|
| 0 % | 0 % | **94 %** | 0,82 | 0 % | 0,78 |
| 10 % (51 tokens) | 9 % | 100 % | 0,80 | **2 %** | 0,78 |
| 25 % (128 tokens) | **24 %** | 100 % | 0,84 | **4 %** | 0,78 |

- **Le recouvrement sert à ne pas couper les phrases** : en découpage fixe, sans
  recouvrement, 6 % des extraits sont coupés en deux (94 % atteignables) ; à 10 %, aucun.
- **La redondance suit le recouvrement nominal en découpage fixe** (9 % et 24 %), au prix
  d'un index plus gros et de doublons dans les résultats.
- **En récursif, le recouvrement n'a presque pas d'effet** (redondance 2 % et 4 %, recall
  identique) : les coupures tombent entre paragraphes ou sections, et
  `RecursiveCharacterTextSplitter` n'applique le recouvrement qu'à l'intérieur d'un
  paragraphe trop long. Le réglage nominal ne dit pas la redondance réelle : il faut la
  mesurer.

##### Sémantique contre récursif : nombre, longueur, cohérence

| Stratégie | Chunks | Tokens moy. | Écart-type | Min-max | Cohésion | Similarité aux coupures |
|---|---|---|---|---|---|---|
| Récursif 512 | 148 | 341 | 147 | 57-552 | 0,653 | 0,417 |
| Titres Markdown | 110 | 451 | 636 | 17-4 405 | 0,646 | **0,354** |
| Sémantique (p95) | 180 | 276 | **357** | **3-3 174** | **0,660** | 0,384 |

- **Cohésion** : similarité moyenne de chaque phrase au centre de son chunk. La sémantique
  est légèrement plus cohésive (0,660 contre 0,653) : ses chunks parlent d'un seul sujet.
- **Similarité aux coupures** : entre la dernière phrase d'un chunk et la première du
  suivant. Plus elle est basse, plus les coupures tombent sur de vrais changements de sujet.
  La sémantique coupe mieux que le récursif (0,384 contre 0,417), mais **les titres font
  encore mieux** (0,354) : sur une encyclopédie, l'auteur a déjà placé les coupures.
- **Mais la sémantique cherche moins bien** (recall 0,68, 6 requêtes perdues pour une
  gagnée) : ses chunks sont très inégaux (de 3 à 3 174 tokens). Les tout petits chunks
  manquent de contexte, les très gros diluent leur sujet.
- Une première mesure de cohérence (similarité entre phrases voisines **à l'intérieur** des
  chunks) donnait 0,44 pour toutes les stratégies : elle ne voit pas les coupures, qui sont
  la seule chose qui diffère. Elle a été remplacée par les deux mesures ci-dessus.
- `langchain-experimental` **n'est plus maintenu** : raison supplémentaire de ne pas
  l'utiliser en production.

##### Late chunking

**Principe.** En découpage classique, chaque chunk est encodé seul : « Elle s'obtient après
5.a4 Ff5 » ne sait pas que « elle » est la variante Alapine de la défense slave. Le late
chunking encode **d'abord le document entier** (chaque token reçoit un vecteur qui tient
compte du contexte), **puis** agrège les vecteurs des tokens de chaque segment.

**Prototype** (`app/rag/eval/late_chunking.py`, `scripts/late_chunking.py`) : mêmes chunks
(récursif 256), mêmes 50 requêtes.

| Variante | recall@5 (50) | MRR |
|---|---|---|
| **Classique (chunk encodé seul)** | **0,80** | **0,62** |
| Late chunking, moyenne des tokens | 0,56 | 0,30 |
| Late chunking, dernier token du segment | 0,64 | 0,45 |

**Explication.** Le late chunking suppose un modèle **bidirectionnel** entraîné à produire
des vecteurs de tokens exploitables (comme jina-embeddings-v3). Qwen3-Embedding est
**causal** et entraîné à résumer un texte entier par son dernier token : ses vecteurs
intermédiaires ne sont pas alignés avec ceux des questions. La variante « dernier token »,
plus fidèle à son entraînement, limite la perte sans l'annuler.
**Conclusion** : le late chunking demanderait de changer de modèle d'embedding ; il n'est
pas retenu.

##### Décision

**Stratégie retenue pour la suite : hiérarchique, enfants récursifs de 256 tokens, parents
= sections plafonnées à 1 024 tokens.**

| Critère | Hiérarchique ≤ 1 024 | Pourquoi |
|---|---|---|
| Recall | 0,86 (meilleur, à égalité avec 1 024) | Les petits enfants rendent la recherche précise |
| Coût | 1 945 tokens pour un top-5 | Le plafond borne ce qui est envoyé à l'agent (budget de 6 000 tokens en phase 6) |
| Robustesse | +4 requêtes, **aucune perdue** | La tendance la plus nette du banc |
| Phase 4 | Le parent est une section : son titre fournit directement le fil d'Ariane | Prépare l'enrichissement |

Écartées : **HTML brut** (phrases cassées), **sémantique** (tailles imprévisibles, recall
plus faible, bibliothèque abandonnée), **late chunking** (incompatible avec le modèle),
**fixe** (coupe les phrases, impose 10 à 25 % de redondance), **1 024 tokens** (coût).
Option de repli simple : **récursif 256** (le moins coûteux : 861 tokens pour un top-5).

**Mise en production** : l'index Milvus (`openings`) reste en récursif 512 pour l'instant.
La stratégie hiérarchique y sera intégrée en **phase 4**, en même temps que l'enrichissement
des chunks (fil d'Ariane, questions hypothétiques, résumé), pour ne réindexer qu'une fois.

**Prudence** : aucun écart avec la référence n'est significatif (p ≥ 0,12 sur 50 requêtes),
hors le HTML brut. La décision repose sur une **tendance cohérente** (recall, recall à budget
égal, aucune requête perdue) et sur le **coût**, pas sur une preuve statistique.

**Tests** : 17 tests des stratégies et du banc (faux tokenizer, faux modèle, faux HTML),
307 tests au total, couverture de branches 98 %.

### Phase 4 : enrichir les chunks et choisir une stratégie par type ✅

| Tâche | Détail |
|---|---|
| **Fil d'Ariane** | En tête de chaque chunk indexé : titre de section et chemin, ex. `Défense sicilienne > Variantes > Najdorf` |
| **Questions hypothétiques + résumé** | Générés par phi4-mini à l'ingestion, une seule fois, puis mis en cache. Gain mesuré en recall@5 |
| **Taille par type de document** | Règle de décision documentée pour **procédure, code source, tableau, courriel**. Dans notre domaine : règlement FFE (procédure), notation PGN et suites de coups (≈ code), tableaux de variantes (tableau). Le courriel est absent du corpus : sa règle est documentée sans données |
| Règle de décision | Une stratégie par type de document, justifiée et documentée ici |
| **Revue par un pair** | Grille de **8 critères** : pertinence de la taille, respect de la structure, autonomie du chunk, qualité des métadonnées, redondance, coût d'indexation, mesurabilité, maintenabilité. Relecteurs : un agent relecteur indépendant et l'auteur du projet |

#### Résultats de la phase 4

Base : stratégie hiérarchique retenue en phase 3 (enfants récursifs de 256 tokens,
parents = sections ≤ 1 024 tokens). Banc et 50 requêtes identiques à la phase 3 ;
résultats dans `data/eval/results/phase4-enrichment.json` (banc) et
`phase4-production.json` (Milvus).

> Les sections 4.1 à 4.5 présentent les mesures **avant la revue par un pair**. La
> section 4.6 donne les constats de la revue, les corrections apportées et les **mesures
> finales**, qui remplacent les précédentes.

##### 4.1 Fil d'Ariane en tête de chaque chunk

Le texte **indexé** de chaque enfant commence par le chemin de sa section ; la section
renvoyée à l'agent est inchangée.

```
Défense slave > Principales variantes > Variante Alapine (ligne principale) > Variante tchèque

Elle s'obtient après les coups 5. a4 Ff5. 5...Ff5 empêche 6. e4.
```

Ce chunk, sans fil d'Ariane, ne dit pas de quelle variante « elle » parle : la question
« Par quels coups obtient-on la variante Alapine ? » ne le trouvait pas dans les 20
premiers résultats ; avec le fil d'Ariane, il est **4ᵉ**.

| | recall@5 (50) | MRR | recall à 2 000 tokens |
|---|---|---|---|
| Sans fil d'Ariane | 0,86 | 0,70 | 0,84 |
| Avec fil d'Ariane | 0,84 | **0,76** | **0,86** |

Effet mesuré requête par requête : **14 requêtes mieux classées, 5 moins bien.** Le fil
d'Ariane rend atteignables des chunks « listes » isolés, mais il rapproche aussi les
chunks d'un même article (ils commencent tous par « Défense scandinave > … »), ce qui
complique le choix du bon chunk entre voisins. Retenu pour le gain de classement (MRR).
Coût : la **redondance** (tokens indexés en double) passe de 1,7 % à 11 %, chaque enfant
répétant son fil d'Ariane. La revue (4.6) montrera que les 4 points perdus sur Réf-30
venaient en réalité d'un problème de notation des coups, pas du fil d'Ariane.

##### 4.2 Questions hypothétiques et résumé

phi4-mini génère pour chaque section, en un appel en sortie structurée, **3 questions**
auxquelles elle répond et un **résumé** de 2 phrases (`app/rag/enrich.py`,
`scripts/generate_enrichment.py`). Chacun devient un vecteur supplémentaire qui renvoie à
la section (index multi-vecteurs). Génération : **127 sections, 381 questions,
127 résumés en 14 min**, tous valides ; mis en cache
(`data/enrichment/phi4-mini.jsonl`, clé = empreinte du texte de la section).

**Contrôle d'un biais** : les questions de Réf-30 ont été générées par phi4-mini à partir
des mêmes passages ; des questions hypothétiques du même modèle pourraient leur
ressembler et gonfler le recall. Le gain est donc mesuré aussi sur les **20 sous-questions
de Multi-10**, rédigées sans phi4-mini.

| Variante (avec fil d'Ariane) | Unités | recall@5 Réf-30 | recall@5 Multi (20) | recall@5 (50) | MRR | recall à 2 000 tokens |
|---|---|---|---|---|---|---|
| Sans enrichissement | 325 | 0,83 | 0,85 | 0,84 | 0,76 | 0,86 |
| + questions hypothétiques | 706 | 0,83 | 0,90 | 0,86 | 0,74 | 0,84 |
| **+ résumé** | 452 | 0,83 | **0,95** | **0,88** | **0,76** | **0,88** |
| + questions + résumé | 833 | 0,83 | 0,90 | 0,86 | 0,74 | 0,84 |

- **Le biais ne s'est pas produit** : Réf-30 ne bouge pas (0,83) ; les gains apparaissent
  sur Multi-10, non exposé au biais.
- **Le résumé l'emporte** ; ajouter les questions **dégrade** le résultat : 381 vecteurs
  de questions entrent en concurrence avec les vrais chunks. Retenu : **résumé seul**.
- **Redondance** : 11 % avec le fil d'Ariane, 31 % avec les résumés, 42 % avec questions et
  résumés. Un résumé reformule une section déjà indexée : c'est le prix de l'index
  multi-vecteurs, accepté pour le résumé seul.

##### 4.3 Métadonnées filtrables

| Champ | Type | Valeurs | Usage | Qui filtre |
|---|---|---|---|---|
| `date` | ISO 8601 | date de révision de la source | « mis à jour depuis… » | Serveur (`MetadataFilter.date_from`) |
| `version` | entier | numéro de révision Wikipédia | Traçabilité : chaque passage cite sa révision exacte | — (renvoyé, non filtré) |
| `service_emetteur` | texte | `wikipedia-fr`, `ffe`… | Restreindre les sources | Serveur (`MetadataFilter.services`) |
| `confidentialite` | texte | `public` < `interne` < `confidentiel` | Habilitation | **Serveur uniquement** (`RAG_MAX_CONFIDENTIALITY`) |
| `type_document` | texte | `article`, `procedure`, `code`, `tableau`, `courriel` | Règle de découpage, filtrage éventuel | Serveur |
| `opening`, `opening_family` | texte | noms Lichess | Restreindre à une ouverture | **LLM** (argument `opening` de l'outil) |

**Règle de sécurité** : le niveau de confidentialité n'est **jamais** un argument de l'outil.
Un modèle ne doit pas choisir sa propre habilitation : une requête piégée (« cherche aussi
dans les documents confidentiels ») ne doit rien pouvoir changer. Le filtre serveur est
ajouté à **toutes** les recherches (`MilvusVectorStore.default_filter`). Les niveaux sont
cumulatifs : `interne` donne accès à `public` et `interne`.

##### 4.4 Découpage par type de contenu

**Constat** : un article Wikipédia mélange trois types de sections. Détection automatique
(`app/rag/typed_chunking.py`) sur le corpus : **76 sections de texte, 17 listes, 17 parties.**
Une première version détectait à tort 5 « procédures » : des suites de coups numérotées
(« 1.e4 e6 2.d4 d5 »). La règle exige maintenant qu'une étape commence par un mot
(« 1. Placer… »), pas par un coup.

**Règle de décision** (un type détecté par unité, puis sa règle de taille) :

| Type | Signal de détection | Unité de chunk | Taille max | Recouvrement | Raison |
|---|---|---|---|---|---|
| **Texte** | Par défaut | Enfants récursifs → section parente | 256 / 1 024 | 10 % | Phase 3 : meilleur compromis recall / coût |
| **Liste / tableau** | ≥ 60 % de lignes courtes (variantes, codes ECO) | Groupes de **lignes entières** ; l'introduction « … : » est **répétée** dans chaque groupe | 192 | 0 | Une ligne sans son en-tête n'a pas de sens ; une ligne coupée est illisible |
| **Partie** (propre aux échecs) | > 50 % de coups, ou titre « Exemple(s) de partie » | **Une unité par partie** (en-tête des joueurs + coups), jamais coupée | 1 024 | 0 | Couper une partie au 23ᵉ coup ne sert à rien |
| **Procédure** | ≥ 3 lignes « 1. Mot… » ou « Étape 2 : » | Groupes d'**étapes entières**, titre répété | 512 | 0 | Une étape coupée devient fausse ; les étapes sont autonomes |
| **Code source** | ≥ 2 lignes `def`, `class`, `import`… | Une **fonction ou classe** entière (séparateurs du langage) | 512 | 0 | L'unité de sens d'un code est la fonction |
| **Courriel** | ≥ 2 en-têtes « De : », « Objet : »… | Le **message seul**, sans citations ni signature | 1 024 | 0 | Les citations dupliquent d'autres messages ; la signature est du bruit |

Les trois derniers types n'existent pas dans le corpus : leurs règles sont vérifiées par
des tests sur de petits exemples **fictifs**, jamais indexés.

| Variante (fil d'Ariane, banc) | Unités | recall@5 Réf-30 | Multi | recall@5 (50) | MRR | recall à 2 000 tokens |
|---|---|---|---|---|---|---|
| Hiérarchique + résumé | 452 | 0,83 | 0,95 | 0,88 | 0,76 | 0,88 |
| **Par type + résumé** | **365** | **0,87** | **0,95** | **0,90** | **0,77** | **0,90** |

Le découpage par type récupère **une question** de Réf-30 (0,83 → 0,87, soit 1 question
sur 30) avec **moins de vecteurs** (les parties ne sont plus morcelées).

**Premier gain statistiquement significatif** : contre le RAG de la phase 2 (récursif 512),
**6 requêtes gagnées, aucune perdue** (McNemar, p = 0,031). Restent manquées : 4 questions
Réf-30 de type « liste » (ref-07, 09, 10, 11) et 1 sous-question de Multi-10.

##### 4.5 Mise en production

- **Code** : les briques partagées (sections, fil d'Ariane, parent/enfant) sont passées de
  `app/rag/eval/` à `app/rag/chunking.py` ; aucun module de production n'importe le code
  d'évaluation. `app/rag/indexing.py` construit les unités (enfants typés + résumés).
- **Milvus** : index **multi-vecteurs**. Chaque unité porte `parent_id`, `parent_text` et
  `unit_kind` ; la recherche récupère 4 fois plus d'unités que demandé (64 au plus), garde
  la meilleure par section et renvoie les sections.
- **Ingestion** (`scripts/ingest_corpus.py`) : résumés lus dans le cache, générés par
  phi4-mini seulement s'ils manquent (`--no-summaries` pour ingérer sans LLM).
  **365 unités, dont 127 résumés, en 48 s.**

**Mesure sur le vrai Milvus** (Réf-30) :

| | Phase 2 | **Phase 4** |
|---|---|---|
| recall@1 | 0,50 | **0,70** |
| recall@5 sans filtre | 0,77 | **0,87** |
| recall@5 avec filtre d'ouverture | 0,80 | **0,93** |
| MRR | 0,60 | **0,76** |

**Seuil de pertinence recalibré : 0,50 → 0,60.** Les scores des bonnes réponses ont monté
(minimum 0,62 au lieu de 0,51). À 0,60, **100 %** des bonnes réponses sont gardées et
**65 %** des questions hors corpus écartées (40 % à 0,50). Le seuil qui maximise
l'exactitude équilibrée (0,70) ferait perdre 17 % de bonnes réponses : écarté, selon la
règle de la phase 2 (ne jamais sacrifier de vraies réponses au seuil).

##### 4.6 Revue par un pair, corrections et mesures finales

**Dispositif.** Un agent relecteur indépendant (sans accès aux choix faits, en lecture
seule) a noté la stratégie sur la grille de 8 critères, en vérifiant les chiffres de la
doc dans le code et les fichiers de résultats. Note : **22 / 32**. L'auteur a ensuite relu
et vérifié chaque constat avant d'agir.

| Critère | Note | Constat du relecteur |
|---|---|---|
| 1. Pertinence de la taille | 3 | Choix mesuré (recall à budget égal), mais des unités « partie » à 1 035 tokens et un top-5 (≈ 1 950 tokens) qui laisse peu de marge pour 3 recherches sous 6 000 tokens |
| 2. Respect de la structure | **2** | **« Partie : jamais coupée » était faux** : 4 sections de parties de plus de 1 024 tokens étaient redécoupées par le séparateur récursif (un parent commençait par « 30... Rh6 31. Dh4+ ») |
| 3. Autonomie du chunk | 3 | Fil d'Ariane et introduction répétée efficaces ; mais 6 enfants de moins de 30 tokens (un fil d'Ariane seul, un titre orphelin) |
| 4. Métadonnées | 3 | Confidentialité filtrée côté serveur : bon point ; identifiants positionnels ; `type_document` non filtrable sur le type réel |
| 5. Redondance | 3 | Pas de doublon dans les résultats, mais la hausse de redondance (1,7 % → 11 % → 31 %) n'était pas documentée |
| 6. Coût d'indexation | 3 | Coût payé une fois et mis en cache ; le gain du résumé tient à 2 requêtes |
| 7. Mesurabilité | **2** | **Risque de sur-ajustement** : une vingtaine de configurations comparées sur les mêmes 50 requêtes, et le seul p < 0,05 vient de cette sélection cumulée |
| 8. Maintenabilité | 3 | Production et évaluation séparées, tests verts ; découpage des parents dupliqué ; pas de test sur une partie de plus de 1 024 tokens |

**Écarts entre la doc et la réalité**, tous vérifiés et reconnus : « jamais coupée »
(faux), « taille max 1 024 » (enfants à 1 035), « récupère les 4 points » (1 seule
question), redondance non mentionnée. Les chiffres de production (0,87 / 0,93, MRR 0,76,
seuil 0,60 → 65 %, p = 0,031) ont été confirmés exacts.

**Corrections apportées** (chacune vérifiée sur le corpus et couverte par des tests) :

| # | Faiblesse | Correction | Effet mesuré |
|---|---|---|---|
| C1 | Parties coupées | `split_games` : **une unité par partie**, détectée par son en-tête (« Karpov-Larsen, Montréal, 1979 ») ; l'introduction « … : » suit la partie, l'analyse qui la commente reste avec elle ; une partie unique > 1 024 tokens est coupée **entre deux coups**, en-tête répété | 45 → **41 parties** entières, plus aucun parent qui commence au milieu d'une partie ; enfant le plus long **682 tokens** (1 035 avant), parent max 990 |
| C2 | Questions sur des listes de coups manquées ; notation différente entre questions (« 1.d4 Cf6 ») et corpus (« 1. d4 Cf6 ») | `normalize_moves` : numéros de coups collés au coup (« 1. e4 » → « 1.e4 », « 4… Cd7 » → « 4...Cd7 »), appliquée **au corpus et aux questions** (`normalize_query`, dans l'outil et le banc) | Les 4 points que le fil d'Ariane semblait perdre sur Réf-30 sont récupérés : **0,83 → 0,87** ; les 4 questions « liste » restent manquées (voir ci-dessous) |
| C3 | Enfants vides de sens | `merge_short` : un enfant de moins de 30 tokens est fusionné avec son voisin | 6 → **3** enfants de moins de 30 tokens (22 tokens au minimum) |
| C4 | Génération fragile (délai dépassé, JSON tronqué par phi4-mini) | Génération bornée (`num_predict` = 400), délai de 120 s, **2 essais puis section sautée et signalée** ; clé de cache insensible aux espaces (C2 ne force pas à tout régénérer) | 1 section sur 174 a échoué au 1er essai et réussi au 2ᵉ |
| C5 | Sur-ajustement aux 50 requêtes | **Holdout-20** : 20 questions rédigées à la main **après le gel** de la configuration, jamais utilisées pour un choix (7 sur des coups, 6 idées, 6 histoire, 1 partie ; 11 articles) | Voir ci-dessous |

**Mesures finales sur le banc** (50 requêtes, remplacent les tableaux 4.1 à 4.4) :

| Variante | Unités | Redondance | recall@5 Réf-30 | Multi (20) | recall@5 (50) | MRR | Tokens top-5 | recall à 2 000 tokens |
|---|---|---|---|---|---|---|---|---|
| Hiérarchique (phase 3) | 322 | 2 % | 0,87 | 0,85 | 0,86 | 0,70 | 1 860 | 0,84 |
| + fil d'Ariane | 322 | 11 % | 0,87 | 0,85 | 0,86 | 0,78 | 1 863 | 0,86 |
| + fil d'Ariane + résumé | 447 | 31 % | 0,87 | 0,95 | 0,90 | **0,80** | 1 890 | 0,90 |
| Par type + fil d'Ariane | 229 | 8 % | 0,87 | 0,85 | 0,86 | 0,78 | **1 593** | 0,86 |
| **Par type + fil d'Ariane + résumé (gelée)** | 367 | 32 % | 0,87 | **0,95** | **0,90** | 0,79 | 1 597 | **0,90** |
| Par type + résumé préfixé du fil d'Ariane | 367 | 34 % | 0,83 | 0,95 | 0,88 | 0,77 | 1 601 | 0,88 |

Enseignements : avec la notation normalisée, le fil d'Ariane ne coûte plus rien sur
Réf-30 et apporte +0,08 de MRR ; préfixer **aussi** les résumés par le fil d'Ariane
(suggestion du relecteur) **dégrade** (−1 question) : un résumé est déjà autonome, le
préfixe le rapproche des enfants voisins. Non retenu.

**Validation sur Holdout-20** (jamais vu, configuration gelée) :

| Configuration | recall@5 | MRR | Tokens top-5 | recall à 2 000 tokens | vs phase 2 |
|---|---|---|---|---|---|
| Phase 2 : récursif 512 | 0,90 | 0,69 | 1 803 | 0,95 | — |
| Phase 3 : hiérarchique | 0,95 | 0,76 | 1 888 | 0,90 | +1 / −0 |
| **Phase 4 : configuration gelée** | **1,00** | **0,88** | **1 640** | **1,00** | **+2 / −0** (p = 0,50) |

Les gains se **confirment sur des questions nouvelles** : même ordre (phase 2 < 3 < 4),
aucune question perdue, MRR de 0,69 à 0,88, 9 % de tokens en moins. Avec 20 questions,
l'écart n'est pas significatif (p = 0,50) : l'objection du relecteur reste valable en
rigueur statistique, mais la **direction** du gain est reproduite sur un jeu indépendant.

**Mesures finales en production** (Milvus, Réf-30, 367 unités dont 138 résumés) :

| | Phase 2 | Phase 4 avant revue | **Phase 4 finale** |
|---|---|---|---|
| recall@1 | 0,50 | 0,70 | **0,73** |
| recall@3 | 0,70 | 0,80 | **0,83** |
| recall@5 sans filtre | 0,77 | 0,87 | **0,87** |
| recall@5 avec filtre d'ouverture | 0,80 | 0,93 | 0,90 |
| MRR | 0,60 | 0,76 | **0,79** |

Seuil 0,60 confirmé (100 % des bonnes réponses gardées, 65 % des questions hors corpus
écartées).

**Toujours manquées : ref-07, 09, 10, 11**, les quatre questions dont la réponse est une
ligne de liste (« 3...dxe4 (variante Rubinstein) »). La normalisation ne suffit pas : une
ligne de 6 mots noyée dans un groupe de 190 tokens pèse peu dans un vecteur. La piste
indiquée par le relecteur, une **recherche lexicale (BM25) fusionnée** avec la recherche
vectorielle pour les requêtes contenant des coups, est notée pour la suite ; elle sort du
cadre de la phase 4.

**Reportés, avec leur phase** :
- **`limit` dépendant du budget de tokens** (le top-5 consomme ≈ 1 600 tokens, 3 recherches
  ≈ 4 800 sur 6 000) : c'est la logique de la boucle de recherche, traitée en **phase 6**
  (budget de 3 recherches et 6 000 tokens).
- Identifiants positionnels (`typed:article:n`) : sans conséquence tant qu'un article est
  toujours réindexé en entier (suppression puis réinsertion), ce qui est le cas.
- `parent_text` copié dans chaque unité : 367 lignes pour 138 parents, négligeable à cette
  échelle ; à revoir si le corpus grossit.
- Découpage des parents dupliqué entre le banc et la production, imports dans les
  fonctions : dette de maintenabilité notée.

**Tests** : 48 tests pour le découpage par type et l'enrichissement (dont 12 issus des
corrections), 365 au total, couverture de branches 98 %. Test d'intégration Milvus
(ingestion réelle, relance sans doublon, filtre) : réussi.

### Phase 5 : retrieval décidé par l'agent + traces ✅

| Tâche | Détail |
|---|---|
| **Flot fixe vs décision de l'agent** | Flot fixe : une recherche à chaque question. Décision de l'agent : il juge si une recherche est utile. Comparaison sur Décision-50 |
| Outil Milvus | Retriever emballé en outil, avec un **schéma d'arguments explicite** (requête, filtres de métadonnées, `limit`) |
| Description d'outil | Gabarit du contrat (`docs/contrat-tools.md` § 2.4) : rôle, quand l'utiliser, exemples, contre-exemples |
| **Journal de décision** | `horodatage, session, étape, choix, arguments, statut, durée, tokens, coût, issue` |
| **OpenTelemetry → Langfuse** | Un span par décision, par appel d'outil et par sous-requête, regroupés par session |
| Coût en local | Tokens d'entrée et de sortie + durée ; 0 € en local, avec une estimation de l'équivalent cloud pour comparer |
| Choix du modèle de l'agent | phi4-mini (sortie structurée), qwen3:1.7b et qwen3:14b (appel natif) comparés sur Tâches-30 |
| Mesures | **Appels d'outil corrects** sur Tâches-30 ; **appels inutiles** sur Décision-50 (les 20 questions hors corpus ne devraient déclencher aucune recherche) |

#### Résultats de la phase 5 ✅

##### 5.0 Le graphe : la position commande, le modèle rédige

Première version du graphe en phase 5 : une boucle unique « le modèle choisit un outil →
exécution → le modèle rechoisit ». Essai réel : qwen3:1.7b répondait de mémoire à une
question d'ouverture sans chercher ; phi4-mini (mode structuré) appelait
`get_opening_theory` trois fois de suite pour un simple « Bonjour ! ». Laisser au modèle
le choix d'appeler Lichess ou Stockfish pour une position est à la fois **inutile** (la
règle est connue : Lichess d'abord, Stockfish hors théorie) et **risqué** (petits modèles).

Graphe retenu (`app/agent/graph.py`), deux entrées :

```
START ─ FEN fournie ?
  ├─ oui ─► inspect ► theory (Lichess) ─ en théorie ?
  │                    ├─ oui ─► quick_eval (Stockfish) ► context (RAG) ► videos ► answer
  │                    └─ non, ou Lichess en panne ─► engine (Stockfish) ► answer
  └─ non (question libre) ─► decide ⇄ tools ► answer
```

| Parcours | Qui décide des outils | Rôle du LLM | Route |
|---|---|---|---|
| Position (FEN fournie) | **Le graphe**, de façon déterministe (énoncé) | Rédiger le conseil à partir des résultats | `POST /api/v1/agent/analyze` |
| Question libre | **Le modèle** (sélection native ou structurée), ou flot fixe | Décider, puis répondre | `POST /api/v1/agent/ask` |

Comportement vérifié par les tests (modèle scripté, Lichess simulé) et en réel :
- en théorie : `inspect → theory → quick_eval (profondeur 12) → context (RAG filtré sur
  le nom Lichess de l'ouverture) → videos → answer` ;
- hors théorie : `inspect → theory → engine (profondeur 18, 3 lignes) → answer`, sans
  contexte ni vidéo (pas de nom d'ouverture) ;
- **Lichess en panne** (503 ×3 tentatives, ou jeton absent → 401) : traité comme hors
  théorie, Stockfish prend le relais ; **FEN invalide** : aucun service externe n'est
  appelé, chaque outil répond `invalid_arguments`, le modèle l'explique ;
- essai réel sans jeton Lichess ni Stockfish installé : les deux outils échouent, la
  réponse dit « les outils Lichess et Stockfish sont introuvables » au lieu d'inventer.

Chaque étape est un span OpenTelemetry (`tool_call`, `decision`, `answer`) ; le journal
de décision et Langfuse en découlent (voir 5.3).

##### 5.1 Choix du modèle de l'agent : taux d'appels d'outil corrects (Tâches-30)

Protocole (`scripts/eval_agent.py --tasks`) : pour chacune des 30 demandes, seule la
**première décision** du modèle est jouée (aucun outil exécuté) ; elle est correcte si
l'outil choisi est l'outil attendu (ou « aucun ») et si les arguments imposés par la
demande sont présents (`lines=3`, `database=lichess`, `depth=35`, FEN exacte…).
Même consigne système et mêmes descriptions d'outils pour les trois configurations.
Résultats dans `data/eval/results/phase5-tool-selection.json`.

| Configuration | Outil correct | Outil + arguments | Latence / décision | Tokens entrée / sortie (30 décisions) |
|---|---|---|---|---|
| phi4-mini, sortie structurée | 73 % (22/30) | 67 % | 1,6 s | 39 149 / 2 484 |
| qwen3:1.7b, appel natif | 70 % (21/30) | 57 % | 0,8 s | 63 055 / 2 521 |
| **qwen3:14b, appel natif** | **97 % (29/30)** | **97 %** | 3,1 s | 63 055 / 1 700 |

Erreurs par configuration :
- **phi4-mini** : appelle un outil quand il ne faut pas (3 fois : « Merci », « capitale
  du Japon », « 64 cases »), et confond `get_opening_theory` avec un autre outil
  (`inspect_position` 2 fois, `evaluate_position` 1, `search_opening_knowledge` 1) ; en
  mode structuré, le modèle semble poussé à toujours remplir `tool`.
- **qwen3:1.7b** : l'inverse, il **n'appelle pas** d'outil 8 fois sur 9 erreurs, surtout
  pour les questions d'idées (`search_opening_knowledge → aucun` : 4 fois) : il répond de
  mémoire malgré la consigne.
- **qwen3:14b** : une seule erreur, « Quel est le meilleur coup pour les Blancs ici ? »
  → `inspect_position` au lieu de `evaluate_position` (la demande est ambiguë : les coups
  légaux sont aussi une réponse).

**Décision : qwen3:14b, appel d'outils natif, pour l'agent** (`AGENT_MODEL=qwen3:14b`,
`AGENT_SELECTION=native`). Le coût est la latence (3 s par décision, sur GPU Mac).
phi4-mini reste le modèle de génération hors agent (résumés d'index, questions de
test), où il n'a pas d'outil à choisir.

Lecture des tokens : en appel natif, les 7 définitions d'outils (descriptions avec
exemples et contre-exemples) coûtent ≈ 2 000 tokens d'entrée par décision ; en mode
structuré, le catalogue est résumé à la première ligne de chaque description (≈ 1 300).
Coût équivalent cloud (hypothèse 0,14 € / 0,55 € par million) : **< 0,01 €** pour les
30 décisions, quelle que soit la configuration ; en local, 0 €.

##### 5.2 Flot fixe ou retrieval décidé par l'agent : taux d'appels inutiles (Décision-50)

Protocole (`scripts/eval_agent.py --decisions`, qwen3:14b, appel natif) : les 50 questions
sont posées à l'agent complet dans les deux modes ; on note si `search_opening_knowledge`
a été appelé. Un appel est **inutile** quand la recherche ne pouvait rien apporter
(20 questions), **oublié** quand elle était utile et n'a pas eu lieu (30 questions).
Résultats dans `data/eval/results/phase5-decision.json`.

| Mode | Appels inutiles (20 questions) | Recherches oubliées (30) | Latence / question |
|---|---|---|---|
| Flot fixe (une recherche à chaque question) | **100 %** (20/20, par construction) | 0 % | 10,4 s |
| **Décision de l'agent** | **20 %** (4/20) | 17 % (5/30) | 13,3 s |

Détail des appels inutiles de l'agent, par catégorie :

| Catégorie (20 questions sans recherche utile) | Appels inutiles | Lecture |
|---|---|---|
| Hors sujet (« capitale de l'Australie », « Bonjour ») | **0 / 6** | Évités |
| Relevant d'un autre outil (« qui est mieux dans cette position ? ») | **0 / 6** | Évités : l'agent a appelé le bon outil |
| Échecs hors corpus (« gambit letton », « règle des 50 coups ») | 4 / 8 | **Non évitables sans connaître le corpus** : la question est bien une question d'ouverture ; la recherche répond `found=false` et l'agent le dit |

Les 5 recherches oubliées sont des questions du corpus auxquelles le modèle a répondu
de mémoire (4) ou via `get_opening_theory` (1, question sur le code ECO).

**Lecture.** Le retrieval décidé par l'agent supprime **tous** les appels évitables
(12/12) au prix de 17 % de recherches oubliées ; le flot fixe ne rate rien mais cherche
toujours, y compris pour « Bonjour ». Le coût réel du flot fixe n'est pas la recherche
(50 à 100 ms) mais la **rédaction** sur des passages inutiles : 10 s par question, parce
que le modèle reçoit jusqu'à 5 sections de contexte même quand elles sont hors sujet.
Le choix dépend du produit : pour un coach où chaque question d'ouverture mérite une
source, le flot fixe est défendable ; pour un assistant qui dialogue aussi (salutations,
positions, vidéos), la décision de l'agent est préférable. **Retenu : décision de
l'agent** pour le parcours « question », les positions suivant de toute façon leur
parcours déterministe (5.0).

##### 5.2 bis Parcours « position » de bout en bout dans Docker

Image reconstruite avec **Stockfish 17.1** (paquet Debian, `/usr/games/stockfish`) ; jeton
Lichess et clé YouTube transmis au conteneur depuis `.env`. Appel réel
`POST /api/v1/agent/analyze` sur l'Italienne après 3.Fc4 :

| Étape | Résultat |
|---|---|
| `inspect_position` | position valide, trait aux Noirs |
| `get_opening_theory` (Lichess, jeton) | **en théorie**, « Italian Game » (C50) : Fc5 25 481 parties, Cf6 21 370, Fe7 1 043 |
| `evaluate_position` (Stockfish 17.1, profondeur 12) | +0,17, « position équilibrée », meilleur coup Cf6 |
| `search_opening_knowledge` (Milvus) | 1 passage sur l'Italienne |
| `find_opening_videos` | `not_configured` (pas de clé YouTube) : signalé, pas bloquant |
| Rédaction (qwen3:14b) | nomme l'ouverture, cite les 3 coups avec leurs effectifs, l'évaluation et le coup de Stockfish, signale l'absence de vidéo |

59 s de bout en bout (encodeur chargé à froid dans le conteneur + rédaction par le
modèle local). Limite observée : dans la rédaction, l'idée attribuée à Fc5 (« développer
la dame ») est **inventée** par le modèle, le contexte RAG (1 passage) ne la contenant
pas. La consigne « n'invente rien » ne suffit pas à un modèle de 14 milliards de
paramètres ; piste pour la suite : ne rédiger l'« idée » d'un coup que si un passage la
documente, sinon se limiter aux statistiques.

**Incident GPU.** La première mesure Multi-10 a été invalidée : 11 recherches sur 15 en
timeout, l'encodeur Qwen3-Embedding sur le GPU du Mac (MPS) étant privé de mémoire par
Ollama (qwen3:14b) et par la construction de l'image Docker en parallèle. Correction :
réglage `EMBEDDING_DEVICE` (`cpu` pour les mesures, l'encodage d'une question courte y
prend moins d'une seconde) et **préchauffage** de l'encodeur au démarrage (chargement +
une phrase encodée) pour que le premier appel ne paie pas le chargement du modèle.
bfloat16 est conservé sur CPU : float32 a été mesuré plus lent sur cette machine.

Incidents corrigés au passage : `uv sync` échouait à la précompilation en bytecode d'une
dépendance (désactivée : `UV_COMPILE_BYTECODE=0`) ; le healthcheck MongoDB (5 s) expirait
quand la machine est chargée, bloquant le démarrage du backend (porté à 15 s).

##### 5.3 Journal de décision et traces OpenTelemetry → Langfuse

**Une seule instrumentation, deux sorties.** Chaque étape de l'agent ouvre un span
OpenTelemetry (`app/agent/tracing.py`) portant des attributs `p13.*`. Un `SpanProcessor`
(`app/agent/journal.py`) transforme les spans terminés en entrées du **journal de
décision** (mémoire + JSON Lines, `data/journal/`) ; le SDK Langfuse (v5, bâti sur
OpenTelemetry) exporte les mêmes spans vers Langfuse quand ses clés sont configurées.
Sans clés, rien ne sort de la machine.

**Schéma d'une entrée du journal :**

| Champ | Contenu | Exemple |
|---|---|---|
| `timestamp` | Début de l'étape, ISO 8601 UTC | `2026-10-22T16:57:05.211+00:00` |
| `session_id` | Session de conversation (regroupement Langfuse) | `partie-1` |
| `trace_id` | Identifiant OpenTelemetry du tour, identique dans Langfuse | `c855e83a…` |
| `step` | Numéro de l'étape dans le tour | `2` |
| `kind` | `decision` (choix du modèle), `tool_call` (exécution), `answer` (rédaction) | `tool_call` |
| `choice` | Outil choisi, ou `answer` / `none` | `search_opening_knowledge` |
| `arguments` | Arguments de l'appel | `{"query": "idées de la Sicilienne"}` |
| `status` | `success`, `error`, `skipped` | `error` |
| `duration_ms` | Durée de l'étape | `134.2` |
| `tokens_in`, `tokens_out` | Tokens du modèle (0 pour un appel d'outil) | `1974`, `65` |
| `cost_eur` | 0 en local ; estimation « équivalent cloud » si un tarif est réglé | `0.000312` |
| `outcome` | Issue courte : code d'erreur, nombre de passages, extrait de réponse | `[not_indexed] collection absente` |

**Pièges rencontrés avec Langfuse auto-hébergé** (6 conteneurs : web, worker,
PostgreSQL, ClickHouse, Redis, MinIO, profil Compose `tracing`) :

| Symptôme | Cause | Correction |
|---|---|---|
| Aucune trace, « Dropping span due to should_export_span filter » (visible seulement avec `LANGFUSE_DEBUG=true`) | le SDK v5 n'exporte par défaut que les spans de son propre traceur et des bibliothèques LLM qu'il connaît ; nos spans `p13.agent` étaient ignorés en silence | `should_export_span=lambda span: True` à la création du client |
| « Failed to export spans batch due to timeout » pendant les mesures, lots de spans perdus | délai d'export OTLP de **5 s** par défaut, dépassé quand la machine est chargée (modèle local, encodeur, ClickHouse) ; un lot en échec n'est pas rejoué | délai porté à 30 s (`langfuse_timeout`) |
| `GET /api/public/traces` et `/sessions` répondent **404** alors que les spans sont bien dans ClickHouse (`events_core`) | Langfuse **v4** fonctionne en mode « events only » : l'ancienne API traces/sessions n'existe plus | lire `GET /api/public/v2/observations?sessionId=…`, qui liste les observations d'une session (traces racines portant `sessionId`, outils, générations) |

Dans Langfuse, les spans portent `langfuse.session.id` (regroupement par session),
`langfuse.trace.name` (`analyze_position`, `agent_turn`, `first_decision`) et le type
d'observation (`tool` pour les appels d'outil). Les générations du modèle (entrées,
sorties, tokens) sont ajoutées par le `CallbackHandler` LangChain de Langfuse.

**Décision (phase 7).** Langfuse reste **optionnel** : son profil Compose n'est pas lancé
par défaut, car ses 6 conteneurs pesaient autant que le reste de la pile pendant les
mesures (ClickHouse à 1,5 à 2 cœurs en continu). Le journal de décision local est la
source de référence des scripts d'évaluation et de la doc ; Langfuse sert à visualiser une
session quand on le démarre (`docker compose --profile tracing up -d`).

**Coût.** Les modèles tournent en local : coût réel 0 €. Le journal calcule un
« équivalent cloud » à partir des tokens (tarif réglable, hypothèse 0,14 € / 0,55 € par
million de tokens d'entrée / sortie), pour donner un ordre de grandeur de ce que
coûterait le même trafic chez un fournisseur.

### Phase 6 : recherche multi-sauts sous budget ✅

| Tâche | Détail |
|---|---|
| Décomposition | Une question multi-sauts devient des **sous-requêtes successives dépendantes** : la 2ᵉ utilise la réponse de la 1ʳᵉ |
| Boucle de recherche | **Critère d'arrêt** (information suffisante ou budget atteint) et **reformulation** de la requête si une recherche ne ramène rien de pertinent |
| **Budget** | **3 recherches** et **6 000 tokens** de contexte récupéré au maximum |
| Comparaison | Multi-sauts vs RAG en un seul passage sur Multi-10 |

#### Résultats de la phase 6 ✅

##### 6.1 La boucle de recherche (`app/agent/multihop.py`)

```
question ──► plan : première sous-question (LLM, sortie structurée)
               │
               ▼
      ┌─► search (search_opening_knowledge, 4 passages) ──► passages retenus
      │        │                                            (dédoublonnés par section,
      │        ▼                                             ajoutés par score tant que
      │     judge (LLM) : assez d'informations ?             le budget le permet)
      │        ├─ oui ────────────────────────────────────► answer
      │        ├─ non, rien trouvé → reformuler ─────┐
      │        └─ non → sous-question suivante ──────┤  (elle dépend de ce qui vient
      └──────────────────────────────────────────────┘   d'être appris)
      arrêts forcés : 3 recherches, ou 6 000 tokens de contexte
```

| Élément | Mise en œuvre |
|---|---|
| **Décomposition en sous-requêtes dépendantes** | `plan` produit la première sous-question ; après chaque recherche, `judge` reçoit la question, les sous-questions déjà posées et les passages retrouvés, et formule la suivante **à partir de ce qui a été trouvé** (ex. le nom « gambit Evans » découvert au saut 1 sert au saut 2) |
| **Critère d'arrêt** | `judge` répond `enough=true` ; ou plus de sous-question possible ; ou garde-fous |
| **Reformulation** | si la dernière recherche n'a rien retourné (`found=false`) ou rien de nouveau, le jugement est invité à **reformuler** avec d'autres mots ; une sous-question identique à une précédente est refusée (elle ne ferait que répéter) |
| **Budget : 3 recherches** | compteur de recherches, arrêt `max_searches` ; aucun jugement n'est demandé après la 3ᵉ (un appel LLM économisé) |
| **Budget : 6 000 tokens de contexte** | les passages d'une recherche sont ajoutés **par score décroissant** tant que le cumul (compté avec le tokenizer du modèle d'embedding) tient ; sinon arrêt `budget`. Un passage déjà retenu n'est jamais compté deux fois. C'est la réponse à la remarque du relecteur de la phase 4 : `limit` est désormais borné par le budget restant |
| **Traces** | chaque sous-question = span `decision` (`plan`, `next_hop`, `reformulate`, `stop`) + span `tool_call` ; une session Langfuse par exécution |

Tests (`tests/agent/test_multihop.py`, modèle scripté, base en stub) : deux sauts
dépendants puis arrêt, plafond de 3 recherches, reformulation après une recherche vide,
budget de contexte qui refuse le 3ᵉ passage, dédoublonnage, sous-question identique,
erreur d'outil tolérée, JSON invalide toléré.

##### 6.2 Multi-sauts contre un seul passage sur Multi-10 (`scripts/eval_multihop.py`)

Protocole : pour chaque question de Multi-10 (deux informations enchaînées, la seconde
dépend de la première), on compare **un seul passage** (la question entière envoyée telle
quelle à `search_opening_knowledge`, 8 passages) et la **boucle multi-sauts** (budget 3
recherches × 4 passages, 6 000 tokens). Modèle qwen3:14b pour planifier, juger et répondre ;
encodeur sur CPU (voir l'incident GPU en 5.2 bis). Session Langfuse
`eval-multi10-qwen3_14b-20261022T175455`, résultats dans
`data/eval/results/phase6-multihop.json` (et `.rows.json` pour le détail par question).

| Méthode | Les 2 sauts trouvés | Rappel des sauts | Réponse (termes clés) | Recherches | Tokens de contexte | Temps |
|---|---|---|---|---|---|---|
| Un passage | 30 % | 0,60 | 57 % | 1 | 1 208 | 13,0 s |
| Multi-sauts | **40 %** | 0,60 | 40 % | 1,3 | 1 113 | 26,9 s |

Motifs d'arrêt de la boucle : `enough` 8, `no_next_question` 2, `max_searches` 0,
`budget` 0 ; 0 reformulation.

Lecture par question (✓ = les deux informations retrouvées) :

| Question | Un passage | Multi-sauts | Ce qui s'est passé |
|---|---|---|---|
| multi-02 (gambit joué par 4.b4 → qui est Evans) | 1/2 | **2/2** | décomposition dépendante réussie : saut 1 « qui a donné son nom au gambit 4.b4 ? » → saut 2 « qui est l'homme qui a donné son nom au gambit Evans ? » |
| multi-10 (gambit du capitaine gallois → idée après 5…Fe7) | 1/2 | **2/2** | idem : le nom « gambit Evans » appris au saut 1 sert au saut 2 |
| multi-03, multi-09 | 2/2 | 2/2 | questions courtes : une recherche suffit aux deux méthodes |
| multi-01 (3…Cf6 dans l'italienne → code ECO) | **2/2** | 0/2 | le plan a réduit la question à « quelle défense répond 3…Cf6 ? », le passage ramené n'était pas celui attendu et le juge a déclaré **« assez »** sans chercher le code ECO ; la question entière, elle, tombait directement sur le bon passage |
| multi-04, multi-05, multi-06, multi-07 | 1/2 | 1/2 | la seconde information n'est trouvée par aucune méthode ; dans 3 cas le juge s'arrête (« assez ») après **une** recherche alors qu'il manque la moitié de la réponse |
| multi-08 (codes ECO E60–E99 → joueurs soviétiques) | 0/2 | 0/2 | la base ne ramène rien pour « défense dont les codes ECO vont de E60 à E99 » ; le juge renvoie `null` au lieu de reformuler (`no_next_question`) |

**Ce que montre la première mesure.**

- La **décomposition en sous-requêtes dépendantes fonctionne** quand la question l'exige
  vraiment (multi-02, multi-10) : la boucle trouve ce que la recherche en un passage rate.
- Le **critère d'arrêt est trop optimiste** : « assez » dans 8 cas sur 10, dont 7 après une
  seule recherche. Le juge voit des passages qui répondent à la sous-question posée et
  conclut que la question initiale est couverte (perte sur multi-01).
- La **reformulation n'a jamais été déclenchée** : quand une recherche ne ramène rien
  (multi-08), le modèle renvoie `null` plutôt qu'une nouvelle formulation, malgré la consigne.
- Le **plan réécrit la question** avant la première recherche et perd des détails qu'elle
  contenait (coups, noms) : sur Réf-30 (phase 7, première mesure), « pourquoi 4.c3 est-il
  une erreur dans la défense des deux cavaliers ? » est devenu une suite de coups de la
  Sicilienne.
- Le **budget n'a jamais été atteint** (1,3 recherche en moyenne, 2 053 tokens au maximum).

**Deux corrections** (`app/agent/multihop.py`, 3 tests ajoutés) :

| Correction | Mise en œuvre |
|---|---|
| **Reformulation imposée** | si le juge répond « pas assez » sans proposer de sous-question, un appel dédié (`REFORMULATE_PROMPT`, schéma qui exige une sous-question) demande une autre formulation ; seule une répétition d'une sous-question déjà posée arrête la boucle (`no_next_question`) |
| **Première recherche = la question telle quelle** | `Budget.first_hop="question"` (défaut) : aucune réécriture avant le premier saut, la phase 4 ayant mesuré un rappel à 5 de 0,87 avec la question brute ; la décomposition intervient au jugement, à partir de ce qui a été trouvé. `first_hop="plan"` conserve l'ancien comportement |

**Seconde mesure** (session `eval-multi10-qwen3_14b-20261022T182137`, conteneurs étrangers
arrêtés, aucun lot de traces perdu) :

| Méthode | Les 2 sauts trouvés | Rappel des sauts | Réponse (termes clés) | Recherches | Reformulations | Tokens de contexte | Temps |
|---|---|---|---|---|---|---|---|
| Un passage | 30 % | 0,60 | 57 % | 1 | — | 1 208 | 12,0 s |
| Multi-sauts, 1ʳᵉ mesure (plan, abandon) | 40 % | 0,60 | 40 % | 1,3 | 0 | 1 113 | 26,9 s |
| Multi-sauts, 2ᵉ mesure (question brute, reformulation imposée) | 30 % | **0,65** | 50 % | 1,7 | 4 | 1 166 | 28,1 s |

Arrêts de la 2ᵉ mesure : `enough` 7, `max_searches` 3, `no_next_question` 0.

- **multi-01** repasse à 2/2 : la question brute tombe sur le bon passage, puis le juge
  demande le code ECO (2 recherches).
- **multi-08** n'est plus abandonné : 3 recherches, 2 reformulations, la défense est-indienne
  est identifiée (1/2) ; les joueurs soviétiques restent introuvables.
- **multi-02** régresse (2/2 → 1/2) : le juge formule le 2ᵉ saut comme « quel métier
  exerçait l'homme… » là où le plan demandait « qui est l'homme… », et deux reformulations
  ne retrouvent pas le passage sur le capitaine. La formulation de la sous-question suivante
  reste le point sensible.
- **multi-10** : la question brute ramène directement le passage de la 2ᵉ information
  (l'idée après 5…Fe7) et le juge s'arrête : réponse juste (100 %), mais l'intermédiaire
  (le gambit Evans) n'est pas « retrouvé » au sens de la métrique.
- Les écarts de « réponse » sur une même recherche (multi-03 : 100 % → 20 %) tiennent à la
  variabilité du rédacteur, pas à la recherche.

**Décision.** La version corrigée est retenue : elle ne renonce plus jamais (0 abandon),
obtient le meilleur rappel des sauts (0,65) et surtout corrige la phase 7 (rappel Réf-30
de 0,67 à 0,87, voir ci-dessous). Le prix est un temps doublé par rapport à la recherche en
un passage (jugement + rédaction par le modèle local). Les garde-fous (3 recherches, 6 000
tokens) sont maintenant sollicités : 3 arrêts `max_searches`, le budget de tokens jamais.

### Phase 7 : évaluer la boucle agentique ✅

| Tâche | Détail |
|---|---|
| Métriques | **Rappel**, **précision**, **nombre moyen de sauts** |
| Évaluation | Sur Réf-30, traces des appels d'outil et des sous-requêtes **par session** dans Langfuse |
| Objectifs | **Rappel > 0,8** et **moins de 3 sauts** par question |

#### Résultats de la phase 7 ✅

##### 7.1 Métriques (`app/rag/eval/agentic.py`, 4 tests)

| Métrique | Définition |
|---|---|
| **Rappel** | part des questions de Réf-30 dont l'extrait attendu figure dans un passage retenu par la boucle |
| **Précision stricte** | moyenne par question de (passages retenus contenant l'extrait) / (passages retenus) ; un seul extrait est attendu, donc plafonnée à 1/n pour n passages retenus |
| **Précision article** | idem en comptant pertinent tout passage du bon article (mesure le « bruit » hors sujet) |
| **Sauts moyens** | nombre moyen de recherches par question (et maximum) |

##### 7.2 Mesure sur Réf-30 (`scripts/eval_agentic_search.py`, qwen3:14b)

| Mesure | 1ʳᵉ (plan + abandon possible) | 2ᵉ (question brute + reformulation imposée) |
|---|---|---|
| **Rappel** | 0,67 | **0,87** |
| Précision stricte | 0,27 | 0,36 |
| Précision article | 0,64 | **0,90** |
| Sauts moyens (max) | 1,17 (3) | **1,20 (3)** |
| Tokens de contexte retenus | 885 | 1 153 |
| Temps par question | 37,5 s (machine saturée) | 26,9 s |
| Arrêts | `enough` 22, `no_next_question` 7, `max_searches` 1 | `enough` 27, `max_searches` 3 |
| Manquées | 10 | 4 : ref-07, ref-09, ref-10, ref-11 |
| Tokens modèle (entrée / sortie) | 74 184 / 7 023 | 83 139 / 4 710 |
| Coût réel / équivalent cloud | 0 € / 0,013 € | 0 € / 0,014 € |
| Session Langfuse | `eval-agentic-qwen3_14b-20261022T180209` | `eval-agentic-qwen3_14b-20261022T182844` |

**Objectifs : rappel 0,87 > 0,8 ✅ ; 1,2 saut en moyenne < 3 ✅.**

Lecture :

- Les 4 questions manquées sont **exactement celles que la recherche simple manquait déjà
  en phase 4** (rappel à 5 de 0,87, mêmes identifiants) : la boucle n'a rien perdu par
  rapport au RAG de référence, et 3 reformulations n'ont pas suffi à ramener ces passages
  (ref-07 : suite de coups de l'est-indienne ; ref-09 et ref-11 : variantes de la française
  nommées dans des chunks de liste ; ref-10 : le juge a dit « assez » sur un passage voisin).
  C'est une limite de l'encodeur et du découpage, pas de la boucle.
- La **précision article de 0,90** montre que les passages retenus viennent presque toujours
  du bon article ; la précision stricte (0,36) est mécaniquement basse puisqu'un seul
  extrait est attendu parmi 4 passages retenus par recherche.
- La 1ʳᵉ mesure illustre l'intérêt d'une évaluation outillée : elle a révélé les deux
  défauts (réécriture de la question, abandon) corrigés en phase 6, et un biais de
  mesure (3 recherches en timeout, machine saturée par des conteneurs étrangers au
  projet et par ClickHouse).

##### 7.3 Traces par session dans Langfuse

Chaque exécution de script est une **session** Langfuse (`eval-agentic-<modèle>-<date>`,
`eval-multi10-…`) ; chaque question est une **trace** `multihop_search` portant
`langfuse.session.id`, avec ses spans `decision:plan`, `decision:next_hop`,
`decision:reformulate`, `decision:stop` (sous-requêtes), `tool_call:search_opening_knowledge`
(appels d'outil) et `answer:answer`, plus les générations `ChatOllama` du callback
LangChain. Vérifié pendant la mesure via `GET /api/public/v2/observations?sessionId=…`
(traces racines de la session) et dans ClickHouse (`events_core`, regroupement par
`session_id`). Après correction du délai d'export (30 s) et arrêt des conteneurs
étrangers : **0 lot de spans perdu** sur les deux dernières sessions, contre 29 sur la
première.

Le même contenu est dans le **journal de décision local**
(`data/journal/<session>.jsonl`, une ligne par étape), qui reste la source de référence
des scripts ; Langfuse est optionnel (voir 5.3).

---

## 4. Correspondance avec les consignes

Chaque consigne est rattachée à la phase qui la traite.

| Consigne | Phase |
|---|---|
| Script d'ingestion et endpoint `/vector-search` (énoncé de l'étape 2) | 2 |
| Distinguer RAG à flot fixe et retrieval déclenché par la décision de l'agent | 5 |
| Emballer un retriever Milvus en outil avec schéma d'arguments explicite | 2, 5 |
| Rédiger la description d'outil et instrumenter chaque décision avec Langfuse | 5 |
| Instrumenter l'agent avec OpenTelemetry et visualiser les traces dans Langfuse | 5 |
| Tracer chaque appel d'outil avec son statut, sa durée et son coût | 5 |
| Mesurer le taux d'appels d'outil corrects sur trente tâches | 1, 5 |
| Définir le schéma d'un journal de décision | 5 |
| Mesurer le taux d'appels inutiles sur cinquante questions dont vingt hors corpus | 1, 5 |
| Décomposer une question multi-sauts en sous-requêtes dépendantes | 6 |
| Boucle de recherche avec critère d'arrêt et reformulation | 6 |
| Plafonner le budget à trois recherches et 6 000 tokens | 6 |
| Traiter dix questions multi-sauts et comparer au RAG en un seul passage | 1, 6 |
| Métriques : rappel, précision, sauts moyens | 7 |
| Jeu de référence de trente questions avec passages attendus annotés | 1 |
| Tracer les appels d'outil et les sous-requêtes dans Langfuse par session | 5, 7 |
| Vérifier un rappel supérieur à 0,8 et moins de trois sauts par question | 7 |
| Titre de section et fil d'Ariane en tête de chaque chunk | 4 |
| Questions hypothétiques et résumé par chunk | 4 |
| Métadonnées filtrables : date, version, service émetteur, confidentialité | 0, 4 |
| Taille de chunk adaptée au type : procédure, code source, tableau, courriel | 4 |
| Stratégie par type de document et règle de décision documentée | 4 |
| Revue de la stratégie de chunking par un pair (grille de 8 critères) | 4 (§ 4.6) |
| Familles de découpage : fixe, récursif, structurel, sémantique, hiérarchique | 3 |
| `RecursiveCharacterTextSplitter` avec séparateurs adaptés au français | 3 |
| `MarkdownHeaderTextSplitter` et découpage par sections HTML | 3 |
| Comparer 256, 512 et 1024 sur le recall@5 du jeu M2 | 3 |
| Faire varier l'overlap de 0 à 25 % et mesurer la redondance | 3 |
| Découpage sémantique par rupture de similarité entre phrases voisines | 3 |
| `SemanticChunker` avec un seuil au percentile 95 | 3 |
| Comparer nombre, longueur et cohérence des chunks face au récursif | 3 |
| Expliquer le late chunking | 3 |

---

## 5. Points ouverts

| Point | Quand le trancher |
|---|---|
| Modèle et mode de l'agent : phi4-mini (sortie structurée, 4/5) ou qwen3 (appel natif, 5/5) | Phase 5, par la mesure sur Tâches-30 |
| Multi-sauts : la formulation de la sous-question suivante par le juge (multi-02 : « quel métier… » rate le passage que « qui est… » trouvait) ; piste : faire lister au plan toutes les informations à trouver et faire cocher chacune par le juge avant « assez » | Après la phase 7, si le mode question est étendu |
| Les 4 questions de Réf-30 manquées par toutes les méthodes (ref-07, 09, 10, 11) : variantes nommées dans des chunks de liste et suite de coups ; piste : index lexical (BM25) en complément du vectoriel | Hors périmètre du projet |
| ~~Faut-il ajouter des documents de chaque type (procédure, code, tableau, courriel) ?~~ Tranché en phase 4 : règles documentées pour les 6 types, appliquées aux 3 types présents dans le corpus (texte, liste, partie), vérifiées par des tests sur exemples fictifs pour les 3 autres, jamais indexés | — |
