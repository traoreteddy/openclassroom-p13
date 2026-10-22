# Fixtures versionnées

Réponses d'API figées, rejouées par `respx` dans les tests : aucun appel réseau.

| Fichier | Requête d'origine |
|---|---|
| `lichess/masters_italian.json` | `GET explorer.lichess.ovh/masters?fen=<Italienne après 3.Fc4>&moves=5&topGames=4` |
| `lichess/lichess_rare.json` | `GET explorer.lichess.ovh/lichess?fen=<position rare>` (peu de parties : hors théorie) |
| `youtube/search_italian.json` | `GET youtube/v3/search?q=partie italienne explication…` |
| `youtube/error_quota.json` | Réponse 403 `quotaExceeded` de l'API YouTube |
| `youtube/error_key_invalid.json` | Réponse 400/403 clé invalide de l'API YouTube |

**Provenance** : construites à partir du format documenté des API (Lichess Opening
Explorer, YouTube Data API v3 `search.list`), car l'Explorer exige un jeton et
YouTube une clé. Pour les remplacer par des réponses réelles :

```bash
LICHESS_TOKEN=... YOUTUBE_API_KEY=... uv run python scripts/record_fixtures.py
```

Toute modification d'un fichier doit être commitée avec les tests qui en dépendent.
