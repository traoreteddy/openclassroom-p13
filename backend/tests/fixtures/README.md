# Fixtures versionnées

Réponses d'API figées, rejouées par `respx` dans les tests : aucun appel réseau.

| Fichier | Requête d'origine |
|---|---|
| `lichess/masters_italian.json` | `GET explorer.lichess.ovh/masters?fen=<Italienne après 3.Fc4>&moves=5&topGames=4` |
| `lichess/lichess_rare.json` | `GET explorer.lichess.ovh/lichess?fen=<position rare>` (peu de parties : hors théorie) |
| `youtube/search_italian.json` | `GET youtube/v3/search?q=partie italienne explication…` (construite, encore utilisée par `scripts/simulation.py`) |
| `youtube/search_italian_fr.json` | **Réponse réelle** : `GET youtube/v3/search?q=partie italienne échecs ouverture explication&relevanceLanguage=fr&safeSearch=strict&videoEmbeddable=true` (5 résultats) |
| `youtube/search_italian_en.json` | **Réponse réelle** : `GET youtube/v3/search?q=Italian Game chess opening tutorial…` (5 résultats) |
| `youtube/videos_italian.json` | **Réponse réelle** : `GET youtube/v3/videos?part=snippet,contentDetails,statistics,status&id=<10 identifiants>` (durée, vues, likes, langue, intégrable) |
| `youtube/error_quota.json` | Réponse 403 `quotaExceeded` de l'API YouTube |
| `youtube/error_key_invalid.json` | Réponse 400/403 clé invalide de l'API YouTube |

**Provenance** : les fixtures Lichess et `youtube/search_italian.json` sont construites à
partir du format documenté des API ; les trois fixtures YouTube « réponse réelle » ont été
enregistrées le 22 octobre 2026 avec une clé d'API. Pour les remplacer par des réponses réelles :

```bash
LICHESS_TOKEN=... YOUTUBE_API_KEY=... uv run python scripts/record_fixtures.py
```

Toute modification d'un fichier doit être commitée avec les tests qui en dépendent.
