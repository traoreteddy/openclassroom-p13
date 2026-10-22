# Outil vidéos : recherche YouTube de vidéos explicatives

Étape « recherche de vidéos explicatives » : pour une ouverture, trouver des vidéos YouTube
pertinentes, les renvoyer avec leurs métadonnées, les afficher avec lien et lecteur, et
les proposer automatiquement dans le parcours de l'agent.

## 1. Ce qui est livré

| Livrable | Où |
|---|---|
| Endpoint `GET /api/v1/videos/{opening}` | `backend/app/api/routes/tools.py` |
| Outil `find_opening_videos` (requêtes combinées, filtre qualité, cache, quota) | `backend/app/tools/videos.py` |
| Cache et compteur de quota (MongoDB, repli mémoire) | `backend/app/tools/video_store.py` |
| Intégration LangGraph : nœud `videos` du parcours « position » | `backend/app/agent/graph.py` |
| Front : carte « Vidéos pour réviser » (lien + lecteur intégré) | `frontend/src/app/features/advice-panel/` |
| Mesure de pertinence sur 10 ouvertures | `backend/scripts/eval_videos.py`, `data/eval/results/videos-10.json` |
| Tests sans réseau sur réponses réelles enregistrées | `backend/tests/tools/test_videos.py`, `test_video_store.py`, `tests/fixtures/youtube/` |

## 2. Chaîne de traitement

```
opening (nom Lichess) ou query (texte libre)
        │
        ▼
 requêtes « intelligentes »          partie italienne échecs ouverture explication   (FR)
 (nom FR + nom EN + mots-clés)       Italian Game chess opening tutorial             (EN)
        │
        ▼  cache ? ──── oui ──► vidéos (0 unité)
        │  quota du jour + 201 > 10 000 − réserve ? ──► quota_exceeded (503, Retry-After)
        ▼
 search.list × 2  (100 unités chacune, safeSearch=strict, videoEmbeddable=true, 5 résultats)
        │  fusion, dédoublonnage des identifiants
        ▼
 videos.list × 1  (1 unité : durée, vues, likes, langue audio, intégrable)
        │
        ▼
 filtre qualité → classement → cache 7 jours → videos[: max_results]
```

**Requêtes.** Le nom Lichess (anglais, ex. « Italian Game: Two Knights Defense ») est
traduit par un dictionnaire de 28 ouvertures courantes (variante exacte, sinon famille
avant le « : ») ; une ouverture inconnue garde son nom anglais avec les mots-clés
français. En langue `fr`, deux requêtes (FR puis EN) ; en `en`, une seule.

**Filtre qualité** (`select_quality`) :

| Règle | Seuil | Pourquoi |
|---|---|---|
| Durée | 3 à 45 min | écarte les shorts et les streams de deux heures |
| Vues | ≥ 1 000 | signal minimal de qualité ; les likes sont renvoyés mais pas filtrés (souvent masqués) |
| Titre | contient un mot de la requête (≥ 4 lettres) ou un mot d'échecs | écarte les hors-sujet (« italienne » suffit, « échecs » aussi) |
| Langue | audio `fr` quand YouTube la connaît ; inconnue = acceptée | public jeune francophone ; la vidéo en hindi des résultats « Italian Game » est écartée |
| Intégrable | `videoEmbeddable=true` dès la recherche | le lecteur intégré doit fonctionner |

Classement : langue demandée d'abord, puis vues décroissantes. `max_results` (1 à 5)
est appliqué après le filtre, le cache garde toute la liste retenue.

**Métadonnées renvoyées** : `video_id`, `title`, `channel`, `url`, `embed_url`
(`youtube-nocookie.com/embed/…`), `thumbnail_url`, `published_at`, `duration_s`, `views`,
`likes`, `language`, plus `queries`, `from_cache`, `quota_used_today`.

## 3. Quotas de l'API YouTube

| Donnée | Valeur |
|---|---|
| Quota gratuit | 10 000 unités par jour, remise à zéro à minuit heure du Pacifique |
| Coût d'une recherche (`search.list`) | 100 unités |
| Coût des détails (`videos.list`, jusqu'à 50 identifiants) | 1 unité |
| Coût d'une ouverture non cachée (2 recherches + détails) | **201 unités** |
| Ouvertures nouvelles par jour (réserve de 500 unités gardée) | **47** |

Trois mécanismes :

1. **Cache** par (requêtes, langue, durée) pendant 7 jours dans MongoDB (`video_cache`,
   index TTL sur `expires_at`) : la même ouverture ne coûte qu'une fois par semaine.
   Le parcours de l'agent et l'endpoint partagent ce cache.
2. **Compteur local** (`youtube_quota`, une ligne par jour Pacifique) : incrémenté à chaque
   appel ; une recherche qui dépasserait `10 000 − réserve` est refusée **avant** l'appel
   (`quota_exceeded`, `Retry-After` = secondes jusqu'à minuit Pacifique). Les lectures en
   cache restent servies.
3. **Erreur Google** (`403 quotaExceeded`) : traduite en `quota_exceeded`, jamais réessayée
   (une tentative sur quota est une unité perdue).

Si MongoDB est indisponible, le magasin se replie sur un cache mémoire et journalise :
une panne de cache ne bloque jamais une recherche. Réglages : `YOUTUBE_QUOTA_DAILY`,
`YOUTUBE_QUOTA_RESERVE`, `YOUTUBE_CACHE_TTL_S`.

## 4. Cas sans résultat et erreurs

| Situation | Outil | Endpoint |
|---|---|---|
| Rien ne passe le filtre | `videos: []` | 200, `videos: []` et `message` explicite |
| Clé absente | `not_configured` | 503 |
| Quota local ou Google | `quota_exceeded` | 503 + `Retry-After` |
| Clé invalide | `auth_required` | 503 |
| Panne / délai | `timeout`, `network_error`, `upstream_error` (2 tentatives, disjoncteur 3 / 60 s) | 503 |
| Ni `opening` ni `query` | `invalid_arguments` | 422 |

Dans le graphe, l'échec de l'outil est mis dans `data.videos.error` et l'agent répond sans
vidéo, en le disant (vérifié en phase 5 : « Aucune vidéo n'a pu être trouvée »).

## 5. Intégration

- **LangGraph** : le nœud `videos` du parcours « position » appelle
  `find_opening_videos(opening=<nom Lichess>, max_results=2)` dès que Lichess a nommé
  l'ouverture ; en mode « question », le modèle peut choisir l'outil (« montre-moi une
  vidéo sur la sicilienne »).
- **Front** : la carte « Vidéos pour réviser » interroge `GET /videos/{famille}` dès que
  la théorie nomme une ouverture (une fois par famille et par session, cache local), et
  affiche miniature, titre, chaîne, durée, vues, le **lien** YouTube et un **lecteur
  intégré** (`youtube-nocookie`) ouvert à la demande. Les vidéos ramenées par l'agent
  servent de repli si la carte est vide.

## 6. Mesure de pertinence (10 ouvertures, 22 octobre 2026)

`uv run python -m scripts.eval_videos`, API réelle, `max_results=3`, langue `fr`.
Critère automatique : le titre nomme l'ouverture (FR ou EN, hors mots génériques).

| Ouverture | Retenues | Nomment l'ouverture | Exemple (1ʳᵉ vidéo) |
|---|---|---|---|
| Italian Game | 3 | 3 | LE PLAN à connaitre dans l'ITALIENNE (433 k vues) |
| Ruy Lopez | 3 | 1 | Les 3 principes de l'ouverture à connaître aux échecs (1,27 M) |
| Sicilian Defense | 2 | 2 | Introduction à la Défense sicilienne (219 k) |
| French Defense | 3 | 3 | Les enjeux de la Française (112 k) |
| Caro-Kann Defense | 3 | 3 | LA CARO KANN, simple, solide et parfaite pour débuter (201 k) |
| Scandinavian Defense | 3 | 3 | Défense scandinave (1) : partie pédagogique (148 k) |
| Queen's Gambit | 3 | 3 | Comment jouer le GAMBIT DAME aux échecs (447 k) |
| Slav Defense | 3 | 3 | Parties pédagogiques, Slave anti pleutres (59 k) |
| King's Indian Defense | 3 | 2 | Ce système FACILE à apprendre avec les Blancs… (182 k) |
| English Opening | 3 | 2 | La MEILLEURE ouverture pour DÉBUTANTS aux échecs (522 k) |

**10/10 ouvertures avec vidéos, 29 vidéos retenues, 86 % nomment l'ouverture, 100 % en
français ; 2 010 unités consommées (20 % du quota du jour).**

Lecture : les 4 vidéos « hors nom » sont des vidéos générales sur les ouvertures, très vues,
que le mot « ouverture » du titre laisse passer. Elles ne sont pas fausses (elles traitent
d'ouvertures) mais moins ciblées. Piste si l'on veut durcir : exiger un mot du **nom**
de l'ouverture dans le titre quand la requête en contient un ; le prix serait des listes
plus courtes (Ruy Lopez tomberait à 1 vidéo). Le choix actuel privilégie « toujours
proposer quelque chose » pour un public débutant.

Vérification de bout en bout dans Docker : `GET /videos/Italian%20Game` renvoie les vidéos
depuis le cache MongoDB partagé avec la mesure (`from_cache: true`), compteur de quota
visible dans la réponse, index TTL présent sur `video_cache`.

## 7. Choix et justifications

| Choix | Justification |
|---|---|
| `httpx` plutôt que `google-api-python-client` (recommandé par l'énoncé) | client asynchrone déjà utilisé par Lichess, même politique de résilience (timeout, 2 tentatives, disjoncteur), simulable avec respx ; le client Google est synchrone et charge un document de découverte. L'API appelée est la même (`search.list`, `videos.list`) ; le changement se limiterait à une classe |
| Deux appels (recherche puis détails) | `search.list` ne donne ni durée, ni vues, ni langue : le filtre qualité exige `videos.list`, qui ne coûte qu'une unité |
| Requêtes FR + EN | les titres français sont le plus souvent trouvés par la requête FR ; la requête EN élargit quand la FR est pauvre ; la langue audio tranche au filtre |
| Dictionnaire FR de 28 noms | couvre le corpus et les familles courantes ; une ouverture inconnue reste cherchable avec son nom anglais |
| Réserve de quota | l'agent doit pouvoir répondre en fin de journée même si l'endpoint a été sollicité |
| Fixtures réelles | les tests portent sur de vraies réponses (10 vidéos, métadonnées complètes) ; `scripts/record_fixtures.py` les ré-enregistre |
| `youtube-nocookie` pour le lecteur | pas de cookie publicitaire pour un public mineur ; l'URL est construite côté front, jamais prise telle quelle de l'API |
