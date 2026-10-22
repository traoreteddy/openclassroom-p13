# Contrat des outils de l'agent

Ce document fixe, pour chaque outil exposé au modèle (LLM) :
**schéma d'entrée, schéma de sortie, effets, idempotence, erreurs et politique de résilience.**

Il sert de référence pour l'implémentation (`backend/app/tools/`), les tests
(`backend/tests/tools/`) et l'évaluation de la sélection d'outils.

| Statut | Signification |
|---|---|
| ✅ | Implémenté et testé |
| 🔌 | Implémenté et testé ; le service réel sera branché à une étape suivante |

---

## 1. Catalogue

| # | Outil | Source | Effet | Idempotent | Statut |
|---|---|---|---|---|---|
| 1 | `inspect_position` | python-chess (local) | aucun | oui | ✅ |
| 2 | `get_opening_theory` | Lichess Explorer | lecture externe | oui | ✅ |
| 3 | `evaluate_position` | Stockfish (local) | aucun | oui | ✅ |
| 4 | `search_opening_knowledge` | Milvus (RAG Wikichess) | lecture externe | oui | 🔌 Milvus à l'étape 2 |
| 5 | `find_opening_videos` | YouTube Data API v3 | lecture externe (quota) | oui | ✅ |
| 6 | `start_deep_analysis` | Stockfish (tâche de fond) | crée une tâche | oui, par clé | ✅ |
| 7 | `get_deep_analysis` | Stockfish (tâche de fond) | aucun | oui | ✅ |

Les outils 6 et 7 sont la version **différée** d'une analyse longue (voir § 6).

---

## 2. Règles de conception d'un outil exposé

### 2.1 Nommage

| Règle | Bon | Mauvais |
|---|---|---|
| `verbe_objet`, en anglais, `snake_case` | `get_opening_theory` | `openingTheory`, `theorie` |
| Le verbe dit l'action réelle : `get_` (lire), `search_` (recherche floue), `evaluate_` (calcul), `start_` / `get_` (tâche différée) | `search_opening_knowledge` | `get_knowledge` (laisse croire à une lecture exacte) |
| L'objet dit le domaine, pas la technologie | `find_opening_videos` | `youtube_api_call` |
| Pas deux noms proches pour deux outils différents | `evaluate_position` / `inspect_position` | `analyze_position` / `analyse_position` |
| Paramètres : noms explicites, unités dans le nom si besoin | `max_results`, `timeout_s` | `n`, `t` |

### 2.2 Granularité

- **Un outil = une question du joueur.** Le modèle choisit entre 5 intentions
  (identifier, théorie, évaluer, expliquer, montrer), pas entre 15 points d'API.
- **Fusionner** les outils qui ne diffèrent que par une option. Un paramètre
  discriminant (`database: masters | lichess`) vaut mieux que deux outils.
- **Séparer** quand l'effet, le coût ou la latence diffèrent nettement. Exemple :
  `evaluate_position` (< 2 s) et `start_deep_analysis` (plusieurs minutes).
- **Ne jamais exposer** au modèle une étape interne : embeddings, construction
  d'URL, mise en cache.

### 2.3 Idempotence

| Outil | Idempotent ? | Pourquoi |
|---|---|---|
| Lectures (`get_`, `search_`, `find_`, `inspect_`, `evaluate_`) | Oui | Aucun état modifié : les rejouer est sans risque, donc le **retry est autorisé** |
| `start_deep_analysis` | Oui, **par clé** `(fen, depth)` | Le même appel renvoie le même `job_id` au lieu de lancer une 2ᵉ analyse |

Règle : **seuls les outils idempotents sont réessayés automatiquement.**

### 2.4 Description d'outil

Chaque description suit le même gabarit :

1. **Ce que fait l'outil**, en une ou deux phrases, avec ce qu'il renvoie.
2. **À utiliser** : la situation, avec au moins un exemple de requête utilisateur.
3. **Ne pas utiliser** : les contre-exemples explicites, chacun avec l'outil à appeler à la place.
4. **Enchaînement** si besoin (ex. « si `in_theory` est false, appeler `evaluate_position` »).

Les descriptions sont en français (langue des requêtes), les noms en anglais.

---

## 3. Enveloppe commune

### 3.1 Résultat (`ToolResult`)

L'exécuteur ne lève jamais d'exception : il renvoie toujours cette enveloppe.

```json
{
  "tool": "get_opening_theory",
  "ok": true,
  "data": { "...": "sortie de l'outil, conforme à son schéma" },
  "error": null,
  "duration_ms": 142.3
}
```

En cas d'échec, `ok` vaut `false`, `data` vaut `null`, et `error` est rempli :

```json
{
  "kind": "input | transient | unavailable",
  "code": "invalid_arguments",
  "message": "Paramètres invalides pour get_opening_theory : fen : position illégale : pas de roi noir.",
  "hint": "Corriger les paramètres puis rappeler l'outil. Exemple valide : {...}",
  "retryable": false,
  "details": {}
}
```

**Message renvoyé au modèle** : `[code] message Conseil : hint`. Il dit ce qui ne va
pas et quoi faire ensuite, pour que le modèle corrige son appel ou continue sans l'outil.

### 3.2 Taxonomie des erreurs

| Famille (`kind`) | Cause | Exemples de `code` | Politique |
|---|---|---|---|
| `input` (erreur de schéma ou de valeur) | L'appel est faux | `invalid_arguments`, `unknown_tool`, `rejected_by_service`, `unknown_job` | **Pas de retry.** Le modèle corrige l'appel grâce au `hint` |
| `transient` (erreur récupérable) | Incident passager | `timeout`, `network_error`, `upstream_error`, `engine_crashed` | **Retry exponentiel** (tenacity), puis `retries_exhausted` |
| `unavailable` (panne durable) | Le service ne répondra pas maintenant | `circuit_open`, `rate_limited`, `auth_required`, `quota_exceeded`, `not_configured`, `retries_exhausted`, `bad_response`, `internal_error` | **Repli ou abandon signalé** : l'agent continue avec les autres outils et le dit à l'utilisateur |

### 3.3 Traduction des réponses HTTP

| Réponse | Erreur |
|---|---|
| Délai dépassé, connexion coupée | `transient` · `timeout` / `network_error` |
| HTTP 5xx | `transient` · `upstream_error` |
| HTTP 429 | `unavailable` · `rate_limited` (avec `retry_after_s`, 60 s par défaut) |
| HTTP 401 / 403 | `unavailable` · `auth_required` |
| HTTP 400 / 404 / 422 | `input` · `rejected_by_service` |
| Corps non JSON | `unavailable` · `bad_response` |

Le **429 n'est pas réessayé** : Lichess demande d'attendre une minute entière.
Réessayer aggraverait le blocage.

### 3.4 Politique de résilience

Ordre d'application : **disjoncteur → timeout → retry**.

| Mécanisme | Rôle | Réglage par défaut |
|---|---|---|
| Timeout par outil | Borne la durée d'une tentative | Voir chaque outil |
| Retry exponentiel avec gigue (tenacity) | Absorbe les erreurs `transient` | 3 tentatives, attente 0,5 s → 8 s max, gigue 0,5 s |
| Disjoncteur | Coupe un service en panne pour ne plus perdre de temps | S'ouvre après 5 échecs consécutifs, retente après 30 s |
| Repli | Réponse dégradée mais utile | Défini par outil (§ 4) |

États du disjoncteur : **fermé** (les appels passent) → **ouvert** (refus immédiat,
`circuit_open`) → **semi-ouvert** (un appel d'essai : s'il réussit, le circuit se
referme ; s'il échoue, il se rouvre).

---

## 4. Fiches des outils

### 4.1 `inspect_position` ✅

**Rôle** : identifier et décrire la position à partir de sa FEN.

**Description donnée au modèle :**
> Identifie une position d'échecs à partir de sa FEN : FEN normalisée, camp au trait,
> numéro du coup, coups légaux (notation SAN), échec, fin de partie, bilan matériel et
> phase de jeu. Calcul local, instantané et sans appel réseau.
>
> À utiliser : en premier, pour vérifier une FEN fournie par l'utilisateur ou savoir
> quels coups sont légaux avant d'en proposer un.
> Exemple : « Est-ce que Cf3 est légal ici ? » → `inspect_position(fen=...)`
>
> Ne pas utiliser : pour savoir quel coup est le meilleur (→ `evaluate_position`) ou ce
> que dit la théorie (→ `get_opening_theory`).

**Entrée**

| Champ | Type | Obligatoire | Contraintes |
|---|---|---|---|
| `fen` | string | oui | FEN valide et position légale (voir § 5) |

**Sortie**

| Champ | Type | Description |
|---|---|---|
| `fen` | string | FEN normalisée |
| `side_to_move` | `white` \| `black` | Camp au trait |
| `fullmove_number` | int | Numéro du coup |
| `legal_moves` | string[] | Coups légaux en SAN |
| `is_check` | bool | Le camp au trait est en échec |
| `status` | `ongoing` \| `checkmate` \| `stalemate` \| `insufficient_material` \| `draw_by_rule` | État de la partie |
| `material` | `{white, black, balance}` | Points de matériel (P=1, C=F=3, T=5, D=9) |
| `phase` | `opening` \| `middlegame_or_endgame` | `opening` jusqu'au 15ᵉ coup |

| Effets | Idempotent | Erreurs | Résilience |
|---|---|---|---|
| Aucun (calcul local) | Oui | `invalid_arguments` | Timeout 1 s, pas de retry, pas de disjoncteur |

---

### 4.2 `get_opening_theory` ✅

**Rôle** : coups théoriques, nom de l'ouverture, parties de référence.

**Description donnée au modèle :**
> Renvoie ce que dit la théorie des ouvertures pour une position : nom et code ECO de
> l'ouverture, coups joués par les joueurs (avec nombre de parties et % de victoires) et,
> en option, des parties de référence. Indique aussi si la position est encore « en
> théorie » (assez de parties connues).
>
> À utiliser : pour proposer les meilleurs coups théoriques, nommer l'ouverture, ou
> savoir si la partie est sortie des sentiers battus.
> Exemple : « Que jouer après 1.e4 e5 2.Cf3 Cc6 3.Fc4 ? » → `get_opening_theory(fen=...)`
> Exemple : « Montre-moi des parties de maîtres » → `get_opening_theory(fen=..., include_games=true)`
>
> Ne pas utiliser : pour une évaluation chiffrée de la position (→ `evaluate_position`),
> ni pour expliquer les idées de l'ouverture (→ `search_opening_knowledge`).
> Si `in_theory` est false, appeler `evaluate_position` pour conseiller un coup.

**Entrée**

| Champ | Type | Défaut | Contraintes |
|---|---|---|---|
| `fen` | string | — | FEN valide |
| `database` | `masters` \| `lichess` | `masters` | `masters` : parties de maîtres 2200+ ; `lichess` : parties en ligne 1600-2500 (blitz, rapide, classique) |
| `max_moves` | int | 5 | 1 à 12 |
| `include_games` | bool | false | Ajoute jusqu'à 4 parties de référence |

**Sortie**

| Champ | Type | Description |
|---|---|---|
| `fen`, `database` | — | Rappel de la requête |
| `opening` | `{eco, name}` ou null | Ex. `{"eco": "C50", "name": "Italian Game"}` |
| `total_games` | int | Parties connues dans cette position |
| `in_theory` | bool | `true` si au moins un coup connu **et** `total_games` ≥ 20 (seuil configurable `THEORY_MIN_GAMES`) |
| `moves[]` | `{san, uci, games, white_pct, draw_pct, black_pct, average_rating}` | Coups triés par popularité |
| `reference_games[]` | `{id, white, black, white_rating, black_rating, year, winner, url}` | Vide si `include_games=false` |

| Effets | Idempotent | Erreurs | Résilience |
|---|---|---|---|
| Lecture de l'API Lichess (jeton `LICHESS_TOKEN` exigé) | Oui | `invalid_arguments`, `timeout`, `network_error`, `upstream_error`, `rate_limited`, `auth_required`, `rejected_by_service`, `bad_response`, `retries_exhausted`, `circuit_open` | Timeout 5 s, 3 tentatives, disjoncteur 5 échecs / 30 s |

**Repli** : si l'outil est indisponible, l'agent passe par la branche Stockfish
(`evaluate_position`) et précise que la théorie n'a pas pu être consultée.

---

### 4.3 `evaluate_position` ✅

**Rôle** : évaluation rapide par Stockfish et meilleurs coups.

**Description donnée au modèle :**
> Évalue une position avec le moteur Stockfish : score en centipions du point de vue
> des Blancs (ou mat en N), appréciation en français (« léger avantage aux Blancs »…)
> et meilleurs coups avec leur suite principale. Réponse en 1 à 2 secondes.
>
> À utiliser : pour dire qui est mieux et de combien, conseiller un coup quand la
> position est sortie de la théorie, ou juger un coup du joueur.
> Exemple : « Qui est mieux ici ? » → `evaluate_position(fen=...)`
> Exemple : « Donne-moi les 3 meilleurs coups » → `evaluate_position(fen=..., lines=3)`
>
> Ne pas utiliser : pour nommer l'ouverture ou connaître les coups des maîtres
> (→ `get_opening_theory`), ni pour une analyse très profonde (→ `start_deep_analysis`).

**Entrée**

| Champ | Type | Défaut | Contraintes |
|---|---|---|---|
| `fen` | string | — | FEN valide |
| `depth` | int | 16 | 8 à 22 |
| `lines` | int | 1 | 1 à 3 (nombre de variantes, « MultiPV ») |

**Sortie**

| Champ | Type | Description |
|---|---|---|
| `fen`, `depth` | — | Rappel |
| `score_cp` | int ou null | Centipions, point de vue des Blancs (+ = Blancs mieux) ; null si mat |
| `mate` | int ou null | Mat en N (+ : les Blancs matent, − : les Noirs matent) |
| `assessment` | string | « Position équilibrée » (≤ 30), « Léger avantage » (≤ 100), « Net avantage » (≤ 300), « Avantage décisif », « Mat en N », « Partie terminée » |
| `lines[]` | `{rank, score_cp, mate, move_san, move_uci, pv_san[]}` | Variantes ; `pv_san` limitée à 8 demi-coups |

| Effets | Idempotent | Erreurs | Résilience |
|---|---|---|---|
| Aucun (moteur local) | Oui | `invalid_arguments`, `timeout`, `engine_crashed`, `engine_not_found`, `retries_exhausted`, `circuit_open` | Timeout 10 s, 2 tentatives (le moteur est redémarré après un crash), disjoncteur 3 échecs / 30 s |

Position terminée (mat, pat) : pas d'erreur, `lines` est vide et `assessment` le dit.

---

### 4.4 `search_opening_knowledge` 🔌

**Rôle** : contexte de l'ouverture (idées, plans, pièges, histoire) par recherche vectorielle.

**Description donnée au modèle :**
> Recherche dans la base de connaissances sur les ouvertures (articles Wikichess
> indexés dans Milvus) les passages les plus proches d'une question : idées, plans
> typiques, pièges, histoire. Renvoie des extraits avec leur source.
>
> À utiliser : pour expliquer **pourquoi** on joue un coup ou une ouverture.
> Exemple : « Quelle est l'idée de la Sicilienne ? » →
> `search_opening_knowledge(query="idées principales de la défense sicilienne", opening="Sicilian Defense")`
>
> Ne pas utiliser : pour obtenir des coups ou des statistiques (→ `get_opening_theory`),
> ni une évaluation (→ `evaluate_position`). Ne pas passer une FEN comme `query`.

**Entrée**

| Champ | Type | Défaut | Contraintes |
|---|---|---|---|
| `query` | string | — | 3 à 300 caractères, question en langage naturel ; une FEN est refusée |
| `opening` | string ou null | null | Filtre sur le nom d'ouverture (celui renvoyé par `get_opening_theory`) |
| `limit` | int | 4 | 1 à 8 |

**Sortie**

| Champ | Type | Description |
|---|---|---|
| `passages[]` | `{text, opening, section, source_url, score}` | Triés par similarité décroissante (cosinus, 0 à 1) |
| `found` | bool | `false` si aucun passage ne dépasse le seuil de pertinence (0,5 par défaut) |

| Effets | Idempotent | Erreurs | Résilience |
|---|---|---|---|
| Lecture de Milvus et calcul d'un embedding | Oui | `invalid_arguments`, `timeout`, `network_error`, `retries_exhausted`, `circuit_open` | Timeout 5 s, 2 tentatives, disjoncteur 5 / 30 s |

**Repli** : `found=false`, et l'agent explique avec les seules données Lichess et Stockfish.

---

### 4.5 `find_opening_videos` ✅

**Rôle** : vidéos explicatives YouTube sur l'ouverture.

**Description donnée au modèle :**
> Cherche sur YouTube des vidéos explicatives sur une ouverture ou une idée d'échecs.
> Renvoie titre, chaîne, lien et miniature. Préférer des requêtes précises (nom de
> l'ouverture + variante + « explication ») et des vidéos courtes ou moyennes.
>
> À utiliser : quand le joueur veut **voir** une explication, ou pour compléter une
> réponse sur une ouverture identifiée.
> Exemple : « Tu as une vidéo sur l'Italienne ? » →
> `find_opening_videos(query="partie italienne explication ouverture", duration="medium")`
>
> Ne pas utiliser : pour une position hors théorie sans nom d'ouverture (les résultats
> seraient hors sujet), ni pour obtenir des coups (→ `get_opening_theory`).
> Chaque appel consomme du quota : ne pas répéter la même recherche.

**Entrée**

| Champ | Type | Défaut | Contraintes |
|---|---|---|---|
| `query` | string | — | 3 à 120 caractères |
| `max_results` | int | 3 | 1 à 5 |
| `duration` | `short` \| `medium` \| `any` | `medium` | `short` < 4 min, `medium` 4 à 20 min |
| `language` | string | `fr` | Code ISO 639-1 |

**Sortie**

| Champ | Type | Description |
|---|---|---|
| `videos[]` | `{video_id, title, channel, url, thumbnail_url, published_at}` | Titres décodés (entités HTML), recherche en mode `safeSearch=strict` (public jeune) |

| Effets | Idempotent | Erreurs | Résilience |
|---|---|---|---|
| Lecture de l'API YouTube : **100 unités de quota par appel** sur 10 000 par jour | Oui | `invalid_arguments`, `not_configured` (pas de clé), `quota_exceeded`, `auth_required`, `timeout`, `network_error`, `upstream_error`, `retries_exhausted`, `circuit_open` | Timeout 5 s, 2 tentatives (jamais sur quota), disjoncteur 3 / 60 s, **cache** des résultats par requête (MongoDB, à l'étape 3) |

**Repli** : liste vide, et l'agent répond sans vidéo.

---

### 4.6 `start_deep_analysis` et `get_deep_analysis` ✅

**Rôle** : analyse Stockfish profonde (profondeur 30 et plus : une à plusieurs minutes),
exécutée en différé avec un ticket de suivi.

**Descriptions données au modèle :**
> `start_deep_analysis` : lance une analyse Stockfish profonde en tâche de fond et
> renvoie immédiatement un `job_id`. Le même appel (même FEN, même profondeur)
> renvoie le même `job_id`. À utiliser seulement si le joueur demande explicitement
> une analyse approfondie. Sinon → `evaluate_position`.
>
> `get_deep_analysis` : renvoie l'état d'une analyse lancée par `start_deep_analysis`
> (`pending`, `running`, `done`, `failed`) et son résultat quand elle est terminée.
> Rappeler après `poll_after_s` secondes tant que le statut n'est pas `done` ou `failed`.

**Entrées**

| Outil | Champ | Type | Défaut | Contraintes |
|---|---|---|---|---|
| `start_deep_analysis` | `fen` | string | — | FEN valide |
| | `depth` | int | 30 | 23 à 40 |
| `get_deep_analysis` | `job_id` | string | — | Identifiant renvoyé par `start_deep_analysis` |

**Sorties**

| Outil | Champs |
|---|---|
| `start_deep_analysis` | `job_id`, `status`, `poll_after_s` |
| `get_deep_analysis` | `job_id`, `status`, `elapsed_s`, `result` (même format que `evaluate_position`, ou null), `error` (ou null) |

| Outil | Effets | Idempotent | Erreurs | Résilience |
|---|---|---|---|---|
| `start_deep_analysis` | Crée une tâche de fond | Oui, par clé `(fen, depth)` | `invalid_arguments`, `engine_not_found` | Timeout 1 s (la soumission ne fait qu'enregistrer la tâche) |
| `get_deep_analysis` | Aucun | Oui | `unknown_job` | Timeout 1 s |

---

## 5. Validation commune de la FEN

Tous les outils qui reçoivent une FEN appliquent la même validation (`app/tools/fen.py`) :

1. Les espaces sont normalisés.
2. **FEN mal formée** → `invalid_arguments`, avec le format attendu et un exemple.
3. **Position illégale** → `invalid_arguments` avec la raison en français : pas de roi
   noir, plus de 8 pions blancs, le camp qui n'a pas le trait est en échec, etc.
4. La FEN est renvoyée sous forme normalisée (6 champs).

---

## 6. Exécution synchrone, asynchrone et différée

| Mode | Principe | Outils | Quand |
|---|---|---|---|
| Synchrone | L'agent attend la réponse avant de continuer | — (aucun : tout est `async`) | — |
| Asynchrone concurrent | Plusieurs appels lancés ensemble (`asyncio.gather`) ; durée totale proche de **l'appel le plus long** au lieu de la somme | Les 5 outils de lecture | Appels indépendants d'un même tour |
| Différé, avec ticket | Soumission immédiate (`job_id`), puis interrogation de l'état | `start_deep_analysis` / `get_deep_analysis` | Traitement plus long que le délai acceptable d'un tour (plus de 10 s) |

**Gain attendu de la concurrence** : en séquentiel, la durée d'un tour est la **somme**
des latences ; en concurrent, c'est la latence de **l'outil le plus lent**. Pour un tour
type (théorie ≈ 300 ms, évaluation ≈ 1 s, connaissances ≈ 200 ms, vidéos ≈ 400 ms),
on passe de ≈ 1,9 s à ≈ 1 s, soit **environ −45 %**. Le gain est borné par l'outil le
plus lent (ici Stockfish) : paralléliser ne raccourcit jamais l'appel le plus long.
Mesure réelle sur 8 appels : voir § 8.

**Limite** : Stockfish n'analyse qu'une position à la fois (verrou dans
`StockfishAnalyzer`). Deux `evaluate_position` lancés ensemble s'exécutent donc l'un
après l'autre ; seuls les appels vers des services différents se parallélisent.

**Dépendances entre outils** : `search_opening_knowledge` et `find_opening_videos`
utilisent le nom de l'ouverture renvoyé par `get_opening_theory`. Le graphe LangGraph
lance donc d'abord `inspect_position` puis `get_opening_theory` et `evaluate_position`
en parallèle, puis les deux suivants en parallèle.

---

## 7. Stratégie de test

### 7.1 Trois niveaux

| Niveau | Ce qu'on vérifie | Doubles de test | Réseau |
|---|---|---|---|
| **Test d'outil** (unitaire) | Contrat d'un outil : schéma, parsing, erreurs, résilience | Mocks HTTP (respx), stubs (moteur, base vectorielle) | Jamais |
| **Test de boucle** (agent) | Le bon outil est choisi, les erreurs sont transmises au modèle, l'agent continue | LLM simulé (réponses scriptées) + outils simulés | Jamais |
| **Test complet** (bout en bout) | Tout le système, vrais services | Aucun | Oui (marqué `integration`, exclu par défaut) |

### 7.2 Choix du double de test

| Technique | Usage dans le projet |
|---|---|
| **Mock** (respx) | API HTTP Lichess et YouTube : on vérifie aussi la requête envoyée (paramètres, en-têtes) |
| **Stub** | Moteur Stockfish, embedder, Milvus : objets factices qui renvoient des réponses prévues |
| **Fixtures versionnées** (« cassettes ») | Réponses JSON réalistes dans `tests/fixtures/`, ré-enregistrables avec `scripts/record_fixtures.py` quand le jeton Lichess et la clé YouTube sont disponibles |
| **Environnement d'intégration** | Docker Compose (Stockfish, Milvus, MongoDB) pour les tests `integration` uniquement |

### 7.3 Objectifs et résultats

| Objectif | Résultat |
|---|---|
| Les 5 outils couverts par pytest **sans appel réseau facturé** | ✅ 183 tests. Un garde-fou (`_no_network` dans `tests/conftest.py`) fait échouer tout test qui tenterait une vraie requête HTTP |
| Cas limites en tests **paramétrés** | ✅ 14 FEN invalides (vide, mal formée, sans roi, 9 pions, triple échec…), bornes de chaque paramètre, champ inconnu, FEN passée comme question, positions terminées |
| API externes simulées avec **respx** et **fixtures versionnées** | ✅ `tests/fixtures/lichess/*.json`, `tests/fixtures/youtube/*.json` |
| Couverture de **branches ≥ 80 %** | ✅ **99 %** (`uv run pytest --cov`, seuil bloquant `fail_under = 80`) |

Lancer les tests :

```bash
cd backend
uv run pytest --cov
```

### 7.4 Ce qu'un test d'outil vérifie

Pour chaque outil :

1. **La requête envoyée** (respx) : paramètres, en-tête d'authentification, filtres.
2. **Le parsing** : la réponse figée donne la sortie attendue (pourcentages, nom ECO, etc.).
3. **Chaque code d'erreur du contrat** : bonne famille, bon code, bon `hint`.
4. **La politique de résilience** : nombre exact de tentatives (`route.call_count`),
   absence de retry sur les erreurs non transitoires, ouverture du disjoncteur.
5. **La validation d'entrée** : un appel invalide n'atteint jamais le service externe.

---

## 8. Mesures

Les deux premières mesures tournent sans réseau, dans un environnement simulé
(`scripts/simulation.py`) : les API sont servies par un `httpx.MockTransport` avec une
latence et un taux de pannes réglables, et le moteur et la base vectorielle sont factices.

| Mesure | Script | Statut |
|---|---|---|
| Gain de latence de la concurrence | `uv run python -m scripts.bench_concurrency` | ✅ Mesuré |
| Effet de la résilience | `uv run python -m scripts.bench_resilience` | ✅ Mesuré |
| Sélection d'outils (30 requêtes, catalogue v0 puis v1) | À écrire | ⏳ **Nécessite un LLM** (fournisseur à choisir) |

### 8.1 Exécution concurrente : 8 appels

Appels : `inspect_position`, `get_opening_theory` ×3 (masters, lichess, avec parties),
`evaluate_position`, `search_opening_knowledge` ×2, `find_opening_videos`.
Latences simulées : Lichess 300 ms, YouTube 400 ms, Stockfish 1 s, Milvus 200 ms.

| Mode | Durée totale (8 appels) |
|---|---|
| Séquentiel (un `await` après l'autre) | 2,72 s |
| Concurrent (`asyncio.gather`) | 1,00 s |

**Gain de latence : −63 % (2,7 fois plus rapide).** Le temps concurrent est égal à
l'appel le plus long (Stockfish, 1 s), et le temps séquentiel à la somme des latences
(2,7 s) : la mesure confirme le modèle du § 6.

### 8.2 Résilience : API simulée à 30 % d'erreurs

300 tours d'agent de 4 outils. Lichess et YouTube échouent aléatoirement dans 30 % des
cas (503, délai dépassé, connexion coupée, en proportions égales). Graine fixe : le
résultat est reproductible.

| Indicateur | Avant (aucune politique) | Après (politiques du contrat) |
|---|---|---|
| Appels réussis (Lichess + YouTube) | 72,0 % | **94,2 %** |
| Tours complets (les 4 outils OK) | 52,0 % | **88,7 %** |
| Requêtes HTTP envoyées | 600 | 784 (+31 %) |

**Lecture** :
- Le résultat suit le calcul théorique. Avec *n* tentatives indépendantes, la
  probabilité d'échec est 0,3ⁿ : Lichess (3 tentatives) donne 1 − 0,3³ = 97,3 %,
  YouTube (2 tentatives) donne 1 − 0,3² = 91 %, soit ≈ 94 % en moyenne.
- **Coût** : 31 % de requêtes en plus. C'est acceptable pour Lichess (gratuit), plus
  discutable pour YouTube (quota). C'est pourquoi YouTube n'a que 2 tentatives.
- Le **disjoncteur ne s'ouvre pas** ici : des erreurs aléatoires à 30 % produisent
  rarement 5 échecs consécutifs après retries. Il sert pour les **pannes durables**
  (service arrêté, jeton invalide), testées dans `test_breaker_opens_after_repeated_outages`.
- Les **11 % de tours incomplets restants** ne bloquent pas l'agent : l'outil en échec
  renvoie un message exploitable, et l'agent répond avec les autres sources (repli).

### 8.3 Catalogue v0 (défaillant) et fusion de 7 outils en 3

Le catalogue v0 reproduit un découpage « un outil par point d'API », avec des noms et
des descriptions ambigus. C'est la référence « avant » de l'évaluation.

| Outils v0 (redondants) | Fusionnés en (v1) | Paramètre discriminant |
|---|---|---|
| `get_moves`, `get_opening_name`, `get_master_games`, `get_online_stats` | `get_opening_theory` | `database: masters \| lichess`, `include_games` |
| `get_eval`, `get_best_move` | `evaluate_position` | `lines` (1 = meilleur coup, 3 = alternatives) |
| `search_wiki` | `search_opening_knowledge` | `opening` (filtre) |

Défauts du v0, corrigés en v1 : noms techniques (`get_eval`), descriptions d'une
ligne sans exemple ni contre-exemple, chevauchements (`get_moves` et `get_best_move`
répondent tous deux à « que jouer ? »).

---

## 9. Structure du code

### 9.1 Arborescence

```
backend/
├── app/
│   ├── api/
│   │   ├── deps/__init__.py      # get_registry : accès au catalogue depuis une route
│   │   └── routes/tools.py       # /moves/{fen}, /evaluate/{fen}, /tools (routes fines)
│   ├── core/config.py            # réglages (.env) : jetons, URL, chemins, seuils
│   └── tools/
│       ├── errors.py             # taxonomie : ToolInputError / ToolTransientError / ToolUnavailableError
│       ├── resilience.py         # ResiliencePolicy, CircuitBreaker, call_with_resilience
│       ├── base.py               # ToolSpec, ToolInput/ToolOutput, ToolResult, ToolRegistry
│       ├── http.py               # get_json : appel HTTP + traduction des erreurs
│       ├── fen.py                # type Fen partagé : validation + normalisation
│       ├── position.py           # outil inspect_position
│       ├── lichess.py            # outil get_opening_theory   (client LichessExplorer)
│       ├── stockfish.py          # outil evaluate_position    (StockfishAnalyzer)
│       ├── deep_analysis.py      # outils start_/get_deep_analysis (DeepAnalysisJobs)
│       ├── knowledge.py          # outil search_opening_knowledge (interfaces Embedder, VectorStore)
│       ├── videos.py             # outil find_opening_videos  (client YouTubeSearch)
│       └── catalog.py            # assemblage : build_registry(settings, dépendances)
├── tests/
│   ├── conftest.py               # doubles de test, garde-fou réseau, fabrique de catalogue
│   ├── fixtures/                 # réponses d'API figées (versionnées) + README de provenance
│   ├── test_tools_routes.py      # routes HTTP
│   └── tools/test_*.py           # un fichier de tests par module d'outil
└── scripts/
    ├── simulation.py             # environnement simulé (API à latence et pannes réglables)
    ├── bench_concurrency.py      # mesure séquentiel / asyncio.gather
    ├── bench_resilience.py       # mesure avant / après résilience (30 % d'erreurs)
    └── record_fixtures.py        # ré-enregistre les fixtures depuis les vraies API
```

### 9.2 Quatre couches

| Couche | Fichiers | Responsabilité | Ne connaît pas |
|---|---|---|---|
| **Contrat** | `ToolInput`, `ToolOutput`, `DESCRIPTION` de chaque module | Ce que le modèle voit : nom, description, schémas | Le réseau, les retries |
| **Exécution** | `base.py`, `resilience.py`, `errors.py` | Valider, protéger (timeout, retry, disjoncteur), envelopper le résultat | Les échecs, Lichess, YouTube |
| **Métier / adaptateurs** | `lichess.py`, `stockfish.py`, `videos.py`, `knowledge.py`, `deep_analysis.py`, `position.py` | Appeler un service et traduire sa réponse dans le schéma de sortie | La politique de retry, FastAPI |
| **Assemblage** | `catalog.py`, `app/main.py`, `api/routes/tools.py` | Créer les dépendances réelles, exposer le catalogue à l'agent et en HTTP | Le détail des outils |

### 9.3 Parcours d'un appel

```
 Agent (LLM) ou route HTTP
        │  registry.execute("get_opening_theory", {"fen": "…"})
        ▼
 ToolRegistry.execute ─────────────────────────────── ne lève jamais d'exception
        │ 1. outil connu ?            sinon → ToolResult(error = unknown_tool)
        │ 2. validation Pydantic      sinon → ToolResult(error = invalid_arguments + exemple)
        ▼
 call_with_resilience
        │ 3. disjoncteur ouvert ?     oui  → circuit_open (sans appeler le service)
        │ 4. tentative sous timeout
        │ 5. erreur transitoire ?     oui  → attente exponentielle, nouvelle tentative
        ▼
 LichessExplorer.theory ──► http.get_json ──► API Lichess
        │ 6. réponse traduite en OpeningTheoryOutput (ou en erreur de la taxonomie)
        ▼
 ToolResult {ok, data | error, duration_ms}  ──► to_model_content() pour le modèle
```

### 9.4 Ajouter un outil

1. Créer `app/tools/<domaine>.py` avec : `DESCRIPTION` (gabarit du § 2.4), un
   `ToolInput` avec un exemple (`json_schema_extra`), un `ToolOutput`, un handler
   `async`, et une fonction `build_<nom>_tool(...) -> ToolSpec`.
2. Lever uniquement des erreurs de la taxonomie (§ 3.2), avec un code documenté.
3. Choisir une `ResiliencePolicy` : retry seulement si l'outil est idempotent.
4. L'enregistrer dans `catalog.py`.
5. Ajouter sa fiche au § 4 et ses tests dans `tests/tools/`. `test_catalog.py` vérifie
   automatiquement le nommage, le gabarit de description, l'exemple d'entrée et la
   cohérence retry / idempotence.

---

## 10. Justification des choix

### 10.1 Contrat et exécution

| Choix | Alternative écartée | Justification |
|---|---|---|
| **Schémas Pydantic** pour l'entrée et la sortie | JSON Schema écrit à la main, dataclasses | Un seul modèle produit à la fois le JSON Schema envoyé au LLM et la validation à l'exécution : ils ne peuvent pas diverger. Messages d'erreur précis par champ |
| **`extra="forbid"`** sur les entrées | Ignorer les champs inconnus | Un champ inventé par le modèle (`moves` au lieu de `max_moves`) est signalé au lieu d'être ignoré en silence |
| **L'exécuteur ne lève jamais d'exception** et renvoie un `ToolResult` | Laisser remonter les exceptions | La boucle d'agent reste simple. L'erreur devient un message que le modèle lit et peut corriger. Un outil en panne ne fait pas planter le tour |
| **Trois familles d'erreurs** | Codes HTTP seuls, ou une seule classe d'erreur | Chaque famille correspond à **une** politique : corriger l'appel, réessayer, ou continuer sans l'outil. Le modèle et le code savent quoi faire sans analyser le message |
| **Message d'erreur `[code] message Conseil : hint`**, avec un exemple valide pour les erreurs de schéma | Trace technique (`ValidationError…`) | Le modèle a besoin de savoir quoi changer. L'exemple valide lui donne directement la forme attendue |
| **`in_theory` calculé par l'outil** (seuil de 20 parties) | Laisser le LLM juger à partir des chiffres | Le passage en branche Stockfish devient déterministe et testable, au lieu de dépendre de l'interprétation du modèle |
| **Validation FEN commune** (`fen.py`), avec la raison en français | Valider dans chaque outil, ou laisser l'API répondre 400 | Une FEN invalide n'atteint jamais un service externe : pas de quota consommé, et un message clair |
| **Noms en anglais, descriptions en français** | Tout en français | Les noms suivent la convention `snake_case` des API et des SDK. Les descriptions sont dans la langue des requêtes des joueurs |
| **Gabarit de description vérifié par des tests** | Relecture manuelle | Une description sans exemple ni contre-exemple fait échouer `test_catalog.py` : la règle ne se dégrade pas avec le temps |

### 10.2 Résilience

| Choix | Alternative écartée | Justification |
|---|---|---|
| **tenacity** pour le retry | Boucle écrite à la main, bibliothèque `backoff` | Standard Python, compatible `async`, backoff exponentiel avec gigue intégré, `sleep` injectable : les tests ne subissent pas les attentes |
| **Disjoncteur maison** (≈ 50 lignes) | `aiobreaker` (2021), `purgatory` | Les bibliothèques disponibles sont peu ou plus maintenues. La version maison est courte, lisible, et testable grâce à une horloge injectable |
| **Ordre disjoncteur → timeout → retry**, et un seul échec compté par appel | Compter chaque tentative | Un appel réessayé 3 fois puis abandonné reflète **une** panne du point de vue de l'utilisateur. On évite d'ouvrir le circuit sur un simple incident passager |
| **Gigue** sur l'attente | Attente fixe | Évite que plusieurs appels en échec ne relancent tous au même moment |
| **Pas de retry sur 429** (et sur le quota YouTube) | Retry exponentiel | Lichess demande d'attendre une minute entière, et le quota YouTube ne revient que le lendemain : réessayer aggraverait la situation |
| **Erreurs de schéma hors disjoncteur** | Les compter comme des pannes | Une FEN invalide est une faute de l'appelant, pas du service : elle ne doit pas couper Lichess pour tout le monde |
| **Retry réservé aux outils idempotents** (vérifié par test) | Retry partout | Réessayer un outil qui crée quelque chose peut le créer deux fois. `start_deep_analysis` est rendu idempotent par une clé `(fen, depth)` |

**Réglages par outil :**

| Outil | Timeout | Tentatives | Disjoncteur | Raison |
|---|---|---|---|---|
| `inspect_position` | 1 s | 1 | non | Calcul local instantané : un échec est un bug, pas un incident |
| `get_opening_theory` | 5 s | 3 | 5 échecs / 30 s | API gratuite, souvent lente aux heures de pointe |
| `evaluate_position` | 10 s | 2 | 3 / 30 s | Profondeur 22 maximum ≈ 2 s ; 2ᵉ tentative après redémarrage du moteur |
| `search_opening_knowledge` | 5 s | 2 | 5 / 30 s | Service interne : une coupure réseau Docker se rattrape vite |
| `find_opening_videos` | 5 s | 2 | 3 / 60 s | Chaque tentative coûte du quota : moins de retries, coupure plus longue |
| `start_/get_deep_analysis` | 1 s | 1 | non | Ne fait qu'enregistrer ou lire une tâche en mémoire |

### 10.3 Dépendances et intégration

| Choix | Alternative écartée | Justification |
|---|---|---|
| **`httpx`** pour les outils | `httpx2` (déjà présent pour `TestClient`) | La consigne impose `respx`, qui ne sait simuler que `httpx`. `httpx2` reste utilisé par le `TestClient` de FastAPI |
| **Un seul `AsyncClient` partagé**, créé au démarrage | Un client par appel | Réutilise les connexions (pool) ; fermé proprement à l'arrêt (`ToolDependencies.aclose`) |
| **Injection de dépendances** (interfaces `Analyzer`, `Embedder`, `VectorStore` ; fabrique `build_registry`) | Clients créés à l'intérieur des outils | Les tests et les mesures fournissent des doubles sans rien modifier. Milvus pourra être branché à l'étape 2 sans toucher à l'outil |
| **python-chess** côté backend | Valider la FEN par expression régulière | Validation complète des règles (rois, pions, roques, échecs impossibles), notation SAN et protocole UCI pour Stockfish dans la même bibliothèque |
| **Stockfish démarré à la première analyse**, protégé par un verrou, redémarré après un crash, jeté après un timeout | Démarrage au lancement de l'API, ou un processus par appel | L'API démarre même sans Stockfish. Le verrou respecte la contrainte UCI (une analyse à la fois). Un moteur interrompu au milieu d'une analyse n'est jamais réutilisé |
| **Tâches différées en mémoire** (`DeepAnalysisJobs`) | Celery, Redis, MongoDB | Suffisant pour un POC en un seul processus. Limite assumée : les tâches sont perdues au redémarrage (voir § 11) |
| **Routes HTTP fines** au-dessus des outils | Logique dupliquée dans les routes | `/moves` et `/evaluate` (étape 1) et l'agent utilisent exactement le même code, les mêmes validations et la même résilience |
| **YouTube en `safeSearch=strict`**, durée `medium` par défaut | Recherche sans filtre | Public jeune (FFE). Une vidéo de 4 à 20 minutes répond mieux à « explique-moi ce coup » qu'un cours de 45 minutes |

### 10.4 Tests

| Choix | Alternative écartée | Justification |
|---|---|---|
| **respx** (mock HTTP) pour Lichess et YouTube | Monkeypatch du client | On vérifie aussi la **requête envoyée** (paramètres, jeton, `safeSearch`) et le **nombre exact de tentatives** |
| **Stubs** pour Stockfish, l'embedder et Milvus | Mocks génériques (`MagicMock`) | Un stub implémente la vraie interface : les tests restent lisibles et cassent si l'interface change |
| **Fixtures JSON versionnées**, construites d'après le format documenté | Enregistrement automatique (VCR) | L'Explorer exige un jeton et YouTube une clé, absents pour l'instant. `record_fixtures.py` permettra de les remplacer par des réponses réelles |
| **Garde-fou réseau global** (`_no_network`) | Faire confiance aux tests | Garantit qu'aucun test n'appelle une API réelle ou facturée, même par erreur |
| **Horloge et `sleep` injectés** | Vraies attentes | Les tests de disjoncteur et de backoff s'exécutent instantanément : 183 tests en ≈ 1 s |
| **Couverture de branches** (et pas seulement de lignes) | Couverture de lignes | Les politiques de résilience sont faites de conditions : il faut vérifier chaque chemin (retry ou non, disjoncteur fermé, ouvert ou semi-ouvert) |

### 10.5 `@tool` (LangChain) ou `ToolSpec` : comparaison et choix

LangGraph s'appuie sur les outils LangChain, déclarés avec le décorateur `@tool`. Le décorateur
déduit le nom de la fonction, la description de la docstring et le schéma des
arguments des annotations de type (ou d'un modèle Pydantic passé dans `args_schema`).
Il recouvre donc en partie notre `ToolSpec`. Trois options ont été comparées.

| Option | Principe |
|---|---|
| **A. `@tool` pur** | Les 7 outils sont écrits directement avec `@tool`. La résilience vient de LangGraph (`RetryPolicy` sur les nœuds, gestion d'erreurs de `ToolNode`) ou est recodée dans chaque outil |
| **B. `ToolSpec` seul** (état actuel) | `ToolSpec` + `ToolRegistry`, sans LangChain |
| **C. `ToolSpec` + adaptateur** (retenue) | `ToolSpec` reste la source de vérité ; un adaptateur génère un outil LangChain (`StructuredTool`) par fiche, pour LangGraph |

#### Avantages et inconvénients

| Option | Avantages | Inconvénients |
|---|---|---|
| **A. `@tool` pur** | Standard de l'écosystème, immédiatement reconnaissable. Peu de code. Intégration directe avec `bind_tools`, `ToolNode` et les traces LangSmith | `RetryPolicy` agit sur un **nœud** entier : un échec YouTube rejoue aussi l'appel Lichess du même nœud. Ni disjoncteur ni timeout par outil : il faudrait les recoder dans les 7 outils. Erreurs brutes ou génériques pour le modèle. Routes FastAPI et tests dépendants de LangChain, dont l'API évolue souvent. Pas de déclaration des effets ni de l'idempotence |
| **B. `ToolSpec` seul** | Couvre toute la résilience demandée (timeout, retry tenacity sélectif, disjoncteur, erreurs exploitables). Indépendant de tout framework : réutilisé par les routes, les tests et les mesures. Contrat vérifié automatiquement (`test_catalog.py`). Un seul modèle Pydantic pour la validation et le schéma | Ne se branche pas seul sur LangGraph : `bind_tools` et `ToolNode` attendent des outils LangChain ; il faudrait réécrire la boucle LLM ↔ outils. Socle maison (≈ 400 lignes) à maintenir et à justifier. Moins standard |
| **C. `ToolSpec` + adaptateur** | Cumule A et B : outils LangChain standards côté LangGraph, résilience par outil côté exécution. Nom, description et schéma écrits une seule fois. Coût faible (≈ 20 lignes, aucun outil modifié). Avec `response_format="content_and_artifact"`, le LLM reçoit le texte et l'application les données structurées (panneau Angular). Couplage à LangChain confiné à un fichier | Une couche de plus à expliquer. Double validation des arguments (LangChain puis registre), sans danger mais redondante. Le message de validation de LangChain remplace le nôtre s'il n'est pas rebranché via `handle_validation_error`. Pas de `@tool` visible dans le code |

#### Comparaison par critère

| Critère | A. `@tool` pur | B. `ToolSpec` seul | C. `ToolSpec` + adaptateur |
|---|---|---|---|
| Branché sur LangGraph (`bind_tools`, `ToolNode`) | ✅ | ❌ | ✅ |
| Timeout **par outil** | ❌ (à recoder ×7) | ✅ | ✅ |
| Retry **par outil**, seulement sur erreurs passagères | ⚠️ par nœud | ✅ | ✅ |
| Disjoncteur | ❌ | ✅ | ✅ |
| Erreurs exploitables par le modèle (famille, code, conseil) | ⚠️ génériques | ✅ | ✅ (avec `handle_validation_error`) |
| Réutilisé par les routes FastAPI | ⚠️ | ✅ | ✅ |
| Tests sans dépendance à LangChain | ❌ | ✅ | ✅ (sauf l'adaptateur) |
| Une seule source pour nom, description et schéma | ✅ | ✅ | ✅ |
| Quantité de code | ✅ faible | ⚠️ socle maison | ⚠️ socle + ≈ 20 lignes |
| Reconnaissable par un évaluateur | ✅ | ⚠️ | ✅ |

#### Décision

**Option C retenue.** Elle conserve la résilience par outil et les tests, qui
répondent directement aux exigences (timeout, tenacity, disjoncteur, messages
exploitables, tests sans réseau), et elle s'intègre à LangGraph de façon standard.
A obligerait à recoder la résilience dans chaque outil ; B seule ne se branche pas sur
LangGraph.

**Répartition des rôles :**

```
 LangGraph  ── décide QUAND appeler un outil (nœuds, état, routage, boucle LLM)
     │  bind_tools / ToolNode
     ▼
 Adaptateur ToolSpec → StructuredTool   (app/agent/langchain_tools.py, étape 3)
     ▼
 ToolRegistry ── garantit que l'appel est FIABLE (validation, timeout, retry, disjoncteur)
     ▼
 Outils (Lichess, Stockfish, Milvus, YouTube…)
```

**Esquisse de l'adaptateur** (étape 3) :

```python
from langchain_core.tools import StructuredTool

def to_langchain_tool(registry: ToolRegistry, spec: ToolSpec) -> StructuredTool:
    async def run(**arguments):
        result = await registry.execute(spec.name, arguments)  # validation + résilience
        return result.to_model_content(), result.data           # texte LLM, données app

    return StructuredTool.from_function(
        coroutine=run,
        name=spec.name,
        description=spec.description,
        args_schema=spec.input_model,           # le même modèle Pydantic
        response_format="content_and_artifact",
        handle_validation_error=...,            # notre message [invalid_arguments] + exemple
    )
```

**Tests prévus pour l'adaptateur :** un outil généré a le même nom, la même
description et le même schéma que sa `ToolSpec` ; un appel invalide renvoie notre
message `[invalid_arguments] … Exemple valide : {…}` ; une panne renvoie le message de
la taxonomie au lieu d'interrompre le graphe.

**Variante écartée :** écrire 7 fonctions décorées `@tool` qui délèguent chacune au
registre. Les `@tool` seraient visibles dans le code, mais signature et docstring
seraient dupliquées pour chaque outil. Cette variante ne serait retenue que si un
`@tool` explicite était exigé.

---

## 11. Limites connues et suites

| Limite | Conséquence | Suite prévue |
|---|---|---|
| Fixtures construites, pas enregistrées | Un changement de format d'API ne serait pas détecté | Lancer `record_fixtures.py` dès que le jeton Lichess et la clé YouTube sont disponibles |
| Base vectorielle non branchée | `search_opening_knowledge` répond `not_configured` dans l'application | Étape 2 : adaptateur Milvus (ses erreurs de connexion devront devenir des `ToolTransientError`) |
| Pas de cache YouTube | Une même recherche consomme du quota à chaque fois | Étape 3 : cache MongoDB par requête |
| Stockfish absent de l'image Docker | `evaluate_position` répond `engine_not_found` dans le conteneur | Étape 6 : installer Stockfish dans le `Dockerfile` |
| Tâches différées en mémoire | Perdues au redémarrage ; un seul processus | Si besoin : stockage MongoDB ou file de tâches |
| Disjoncteur par processus | Plusieurs instances de l'API ne partagent pas l'état du circuit | Acceptable pour le POC (une instance) |
| Outils pas encore exposés à LangGraph | L'agent ne peut pas encore les appeler | Étape 3 : adaptateur `ToolSpec` → `StructuredTool` et ses tests (§ 10.5) |
| Taux de sélection non mesuré | Pas encore de preuve chiffrée de la qualité des descriptions | Écrire le jeu de 30 requêtes et l'évaluation v0 / v1 une fois le LLM choisi |
