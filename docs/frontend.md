# Front Angular : échiquier et panneau de conseils de l'agent

Étape 4 du processus (`docs/processus.md`). L'interface relie l'échiquier `ngx-chess-board`
aux sept outils de l'agent à travers l'API `/api/v1`.

## 1. Architecture

```
ChessBoard (ngx-chess-board) ──► GameStateService (chess.js : FEN, légalité, historique)
                                        │  signal fen()
                                        ▼
                                  AdviceService ──────────────► ApiService (HttpClient)
                                        │                            │
                                        ▼                            ▼
                                  AdvicePanel                   /api/v1 (FastAPI)
```

| Fichier | Rôle |
|---|---|
| `core/game-state.service.ts` | **Seule source de vérité** de la partie : FEN courante, coups joués, statut. L'échiquier affiche et saisit ; chess.js valide |
| `core/api.models.ts` | Types des réponses de l'API, miroir des modèles Pydantic du backend |
| `core/api.service.ts` | Une méthode par route : `theory`, `evaluate`, `analyze`, `ask`, `startDeepAnalysis`, `getDeepAnalysis` ; `describeError` traduit une erreur HTTP en message |
| `core/advice.service.ts` | Orchestration : quand appeler quoi, annulation, cache, états de chargement et d'erreur |
| `features/advice-panel/` | Affichage : position, coups joués, coups théoriques, conseil, analyse profonde, questions |
| `proxy.conf.json` | En développement, `ng serve` relaie `/api` vers le backend (port 8001 du `docker compose`) : pas de CORS à gérer dans le navigateur. Le backend accepte aussi l'origine `http://localhost:4200` (`CORS_ORIGINS`) |

## 2. Ce que le panneau montre, et quel outil est derrière

| Carte | Outil / route | Déclenchement |
|---|---|---|
| Coups théoriques (nom ECO, coups des maîtres avec effectifs et résultats, badge en / hors théorie) | `get_opening_theory` · `GET /moves/{fen}` | **automatique** à chaque position, 400 ms après le dernier coup |
| Conseil de l'agent : réponse rédigée, évaluation et lignes Stockfish, passages de la base de connaissances avec leur source | parcours « position » du graphe : `inspect_position` → `get_opening_theory` → `evaluate_position` → `search_opening_knowledge` → `find_opening_videos` · `POST /agent/analyze` | **sur demande** (bouton), avec une question facultative sur la position |
| Vidéos pour réviser (lien et lecteur intégré) | `find_opening_videos` · `GET /videos/{opening}` | **automatique** dès que la théorie nomme une ouverture, une fois par famille |
| Analyse profonde (profondeur 28, une à plusieurs minutes) | `start_deep_analysis` + `get_deep_analysis` · `POST /analysis/{fen}` puis `GET /analysis/{job_id}` | sur demande ; état interrogé toutes les 2 s jusqu'à `done` |
| Poser une question | graphe « question » (le modèle choisit ses outils) · `POST /agent/ask` | sur demande, conversation dans la même session |

Les coups proposés (théorie, lignes du moteur) sont **cliquables** : ils se jouent sur
l'échiquier, ce qui relance la théorie sur la nouvelle position.

## 3. Points de vigilance de l'énoncé

**Chargement et erreurs.** Chaque carte a son indicateur (spinner) et son message d'erreur,
indépendants : une panne de Lichess n'empêche ni de jouer ni de demander un conseil. Les
messages reprennent `message` et `hint` renvoyés par l'API (`{"error": {...}}`), avec un
texte dédié quand le serveur est injoignable (statut 0). La théorie a un bouton
« Réessayer ». L'échiquier n'est jamais bloqué.

**Synchronisation échiquier ↔ agent.**

- Un seul état de position (`GameStateService.fen`), lu par tous.
- `switchMap` : un coup joué pendant un appel de théorie **annule** la requête en cours
  (visible dans le test « annule la requête en cours quand la position change »).
- Une réponse de théorie qui arrive pour une autre position est ignorée.
- Le conseil de l'agent garde la FEN analysée ; si la position bouge, il est marqué
  « dépassé » et ses coups deviennent inactifs, plutôt que d'annuler un appel qui coûte
  20 à 60 s de modèle local.
- L'analyse profonde en cours est abandonnée côté front dès que la position change (le
  serveur termine sa tâche seul, idempotente par FEN et profondeur).

**Moins d'appels inutiles.**

- Délai de 400 ms après le dernier changement de position (`debounceTime`) : jouer vite
  plusieurs coups ne déclenche qu'un appel.
- `distinctUntilChanged` : une position identique n'est pas redemandée.
- **Cache par FEN** des coups théoriques (revenir en arrière avec « Annuler » ne rappelle
  pas Lichess), alimenté aussi par la théorie contenue dans la réponse de l'agent.
- Un conseil déjà en cours pour la même position n'est pas relancé ; une analyse
  profonde identique (même FEN, même profondeur) non plus.
- Le conseil complet et l'analyse profonde sont **à la demande**, jamais automatiques.

## 4. Vidéos

La carte « Vidéos pour réviser » appelle `GET /videos/{famille}` dès que la théorie nomme
une ouverture (famille = nom Lichess avant le « : »), une fois par famille et par session ;
le backend garde ses résultats 7 jours (voir `docs/outil-videos.md`). Chaque vidéo montre
miniature, titre, chaîne, durée, vues, le **lien** YouTube et un **lecteur intégré** sans
cookies (`youtube-nocookie.com/embed/<id>`, URL construite côté front, jamais reprise de
l'API) ouvert à la demande. Les vidéos ramenées par l'agent servent de repli si la carte
est vide. Sans clé YouTube ou quota épuisé, la carte affiche le message de l'API sans
bloquer le reste.

## 5. Tests (Vitest, `npm test`)

| Fichier | Ce qui est vérifié |
|---|---|
| `api.service.spec.ts` | Encodage de la FEN dans l'URL (espaces seulement), paramètres et corps envoyés, messages d'erreur |
| `advice.service.spec.ts` | Chargement puis cache de la théorie, annulation sur changement de position, erreur réseau et réessai, conseil avec question et marquage « dépassé », pas de double appel, suivi d'une analyse profonde jusqu'à `done`, abandon si la position change, conversation |
| `game-state.service.spec.ts`, `app.spec.ts` | Logique de partie ; rendu de la page avec l'échiquier et les quatre cartes |

L'API est simulée avec `HttpTestingController` : aucun test ne touche le réseau.

## 6. Lancer

```bash
docker compose up -d            # API sur http://localhost:8001 (Stockfish, Milvus, MongoDB)
cd frontend && npm start        # http://localhost:4200, /api relayé vers le backend
```
