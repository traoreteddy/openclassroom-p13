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

### Phase 1 : jeux de référence ⏳

| Jeu | Contenu | Sert à |
|---|---|---|
| **Réf-30** (« jeu M2 ») | 30 questions + extraits attendus annotés | recall@5 des découpages ; rappel et précision de l'agent |
| **Décision-50** | 50 questions, dont **20 hors corpus** | Taux d'appels inutiles |
| **Tâches-30** | 30 tâches avec l'outil attendu | Taux d'appels d'outil corrects |
| **Multi-10** | 10 questions multi-sauts avec leurs sous-questions attendues | Multi-sauts vs RAG en un seul passage |

### Phase 2 : RAG simple, la référence (livrables de l'énoncé) ⏳

| Tâche | Détail |
|---|---|
| Script d'ingestion | `scripts/ingest_corpus.py` : nettoyage, découpage récursif 512 tokens, embeddings, chargement dans Milvus (identifiants stables : relançable sans doublons) et textes bruts dans MongoDB. Exécuté dans un conteneur |
| Outil | `search_opening_knowledge` branché sur Milvus et le vrai modèle (il répond aujourd'hui `not_configured`) |
| Endpoint | `GET /api/v1/vector-search?query=…&opening=…&limit=…` |
| Tests | Sans réseau (découpage, adaptateur Milvus, endpoint, ingestion) + test d'intégration avec le vrai Milvus |
| Mesure de référence | recall@5 sur Réf-30 |

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
