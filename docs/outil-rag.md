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
| Traces | **Langfuse auto-hébergé** (conteneurs) + OpenTelemetry | Local, et OpenTelemetry est le standard d'instrumentation |
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

**Étape 2.9 : exécution dans Docker** 🚧. Image reconstruite (2,67 Go ; un premier échec
de `uv sync` était passager, dû à un téléchargement interrompu). Ingestion lancée **dans le
conteneur** : modèle téléchargé une fois dans le volume `hf_models` (1,1 Go), embeddings
calculés **sur CPU** (7 cœurs à 100 %), car Docker n'a pas accès au GPU du Mac. Durée :
plus de 15 minutes, contre 30 s sur l'hôte avec le GPU. Résultat final et appel de
`/vector-search` depuis l'hôte : à compléter.

**Décision : seuil 0,50.** Il garde toutes les bonnes réponses et écarte toutes les
questions hors sujet. Les questions d'échecs hors corpus ou relevant d'un autre outil ont
des scores aussi élevés que les vraies réponses (même vocabulaire) : **aucun seuil ne les
sépare** sans perdre de bonnes réponses. Les éviter relève de la **décision de l'agent**
(phase 5), pas du seuil.

### Phase 3 : comparer les découpages ⏳

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

### Phase 4 : enrichir les chunks et choisir une stratégie par type ⏳

| Tâche | Détail |
|---|---|
| **Fil d'Ariane** | En tête de chaque chunk indexé : titre de section et chemin, ex. `Défense sicilienne > Variantes > Najdorf` |
| **Questions hypothétiques + résumé** | Générés par phi4-mini à l'ingestion, une seule fois, puis mis en cache. Gain mesuré en recall@5 |
| **Taille par type de document** | Règle de décision documentée pour **procédure, code source, tableau, courriel**. Dans notre domaine : règlement FFE (procédure), notation PGN et suites de coups (≈ code), tableaux de variantes (tableau). Le courriel est absent du corpus : sa règle est documentée sans données |
| Règle de décision | Une stratégie par type de document, justifiée et documentée ici |
| **Revue par un pair** | Grille de **8 critères** : pertinence de la taille, respect de la structure, autonomie du chunk, qualité des métadonnées, redondance, coût d'indexation, mesurabilité, maintenabilité. Relecteurs : un agent relecteur indépendant et l'auteur du projet |

### Phase 5 : retrieval décidé par l'agent + traces ⏳

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

### Phase 6 : recherche multi-sauts sous budget ⏳

| Tâche | Détail |
|---|---|
| Décomposition | Une question multi-sauts devient des **sous-requêtes successives dépendantes** : la 2ᵉ utilise la réponse de la 1ʳᵉ |
| Boucle de recherche | **Critère d'arrêt** (information suffisante ou budget atteint) et **reformulation** de la requête si une recherche ne ramène rien de pertinent |
| **Budget** | **3 recherches** et **6 000 tokens** de contexte récupéré au maximum |
| Comparaison | Multi-sauts vs RAG en un seul passage sur Multi-10 |

### Phase 7 : évaluer la boucle agentique ⏳

| Tâche | Détail |
|---|---|
| Métriques | **Rappel**, **précision**, **nombre moyen de sauts** |
| Évaluation | Sur Réf-30, traces des appels d'outil et des sous-requêtes **par session** dans Langfuse |
| Objectifs | **Rappel > 0,8** et **moins de 3 sauts** par question |

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
| Revue de la stratégie de chunking par un pair (grille de 8 critères) | 4 |
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
| Faut-il ajouter quelques documents de chaque type (procédure, code, tableau, courriel), marqués comme exemples, pour tester la règle par type ? | Phase 4 |
