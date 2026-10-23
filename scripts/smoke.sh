#!/usr/bin/env bash
# Vérification de bout en bout de l'application en conteneurs : chaque service est appelé
# à travers le front (nginx → backend → MongoDB, Milvus, Stockfish, Ollama, Lichess, YouTube).
#
#   ./scripts/smoke.sh                 # http://localhost:8080
#   BASE=http://localhost:8090 ./scripts/smoke.sh
#
# Sort avec le code 1 si une vérification échoue. Le conseil de l'agent prend 20 à 90 s.

set -u
BASE="${BASE:-http://localhost:${FRONTEND_PORT:-8080}}"
API="$BASE/api/v1"
ITALIAN="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R%20b%20KQkq%20-%203%203"
RARE="rnbqkbn1/ppppppp1/7r/7p/P7/6R1/1PPPPPPP/1NBQKBNR%20b%20Kq%20-%203%203"  # 1.a4 h5 2.Ta3 Th6 3.Tg3 : hors théorie, position légale
failures=0

check() {  # check <libellé> <corps de réponse> <python qui lit stdin et affiche un résumé ou lève>
  local label="$1" body="$2" summary
  if summary=$(printf '%s' "$body" | python3 -c "$3" 2>&1); then
    printf '  ✓ %-42s %s\n' "$label" "$summary"
  else
    printf '  ✗ %-42s %s\n' "$label" "$(printf '%s' "$summary" | tail -1)"
    printf '      réponse : %s\n' "$(printf '%s' "$body" | tr '\n' ' ' | cut -c1-200)"
    failures=$((failures + 1))
  fi
}

# Les réponses d'erreur (4xx, 5xx) sont renvoyées telles quelles : le JSON {"error": …}
# ou le code HTTP apparaît alors dans la ligne « réponse ».
get() { curl -s -m "${2:-30}" -w ' [HTTP %{http_code}]' "$1" | sed 's/ \[HTTP 200\]$//'; }
post() { curl -s -m "${3:-300}" -w ' [HTTP %{http_code}]' -H 'Content-Type: application/json' -d "$2" "$1" | sed 's/ \[HTTP 200\]$//'; }

echo "Vérification de bout en bout sur $BASE"

check "front : page servie" "$(curl -s -m 10 "$BASE/")" \
  'import sys; s=sys.stdin.read(); assert "<app-root" in s, "pas d’app-root"; print("index.html")'

check "backend : healthcheck" "$(get "$API/healthcheck")" \
  'import sys,json; d=json.load(sys.stdin); assert d["status"]=="ok"; print(d["environment"])'

check "MongoDB : healthcheck/database" "$(get "$API/healthcheck/database")" \
  'import sys,json; d=json.load(sys.stdin); assert d["status"]=="ok"; print("base", d["database"])'

check "Lichess : coups théoriques (Italienne)" "$(get "$API/moves/$ITALIAN?max_moves=3")" \
  'import sys,json; d=json.load(sys.stdin); assert d["in_theory"] and d["moves"], d; print(d["opening"]["name"], "·", d["moves"][0]["san"], d["moves"][0]["games"], "parties")'

check "Stockfish : évaluation (profondeur 12)" "$(get "$API/evaluate/$RARE?depth=12" 60)" \
  'import sys,json; d=json.load(sys.stdin); assert d["lines"], d; print("score", d["score_cp"], "·", d["assessment"], "·", d["lines"][0]["move_san"])'

check "Milvus : recherche vectorielle" "$(get "$API/vector-search?query=id%C3%A9es%20de%20la%20d%C3%A9fense%20sicilienne&limit=2" 120)" \
  'import sys,json; d=json.load(sys.stdin); assert d["found"] and d["passages"], d; print(len(d["passages"]), "passage(s) ·", d["passages"][0]["title"])'

check "YouTube : vidéos (Italian Game)" "$(get "$API/videos/Italian%20Game?max_results=2" 60)" \
  'import sys,json; d=json.load(sys.stdin)
if "error" in d: print("non disponible :", d["error"]["code"], "(toléré)")
else: assert d.get("videos") is not None; print(len(d["videos"]), "vidéo(s)", "(cache)" if d.get("from_cache") else "", d.get("message",""))'

ticket=$(curl -s -m 30 -X POST "$API/analysis/$ITALIAN?depth=23")
job=$(printf '%s' "$ticket" | python3 -c 'import sys,json; print(json.load(sys.stdin)["job_id"])' 2>/dev/null || echo "")
state=""
for _ in $(seq 1 60); do
  state=$(get "$API/analysis/$job" 10)
  printf '%s' "$state" | grep -q '"status":"done"' && break
  printf '%s' "$state" | grep -q '"status":"failed"' && break
  sleep 2
done
check "Stockfish : analyse profonde différée" "$state" \
  'import sys,json; d=json.load(sys.stdin); assert d["status"]=="done", d; print("profondeur", d["result"]["depth"], "en", round(d["elapsed_s"]), "s ·", d["result"]["lines"][0]["move_san"])'

check "Agent : conseil complet (Lichess → RAG → vidéos → rédaction)" \
  "$(post "$API/agent/analyze" "{\"fen\": \"$(printf '%s' "$ITALIAN" | sed 's/%20/ /g')\", \"session_id\": \"smoke\"}")" \
  'import sys,json; d=json.load(sys.stdin); assert d["answer"].strip(), d; tools=[c["name"] for c in d["tool_calls"]]; print(len(d["answer"]), "caractères ·", ", ".join(tools))'

check "Agent : question libre (outils choisis par le modèle)" \
  "$(post "$API/agent/ask" '{"question": "Quelles sont les idées de la défense Caro-Kann ?", "session_id": "smoke"}')" \
  'import sys,json; d=json.load(sys.stdin); assert d["answer"].strip(), d; print(len(d["answer"]), "caractères ·", ", ".join(c["name"] for c in d["tool_calls"]) or "aucun outil")'

echo
if [ "$failures" -eq 0 ]; then
  echo "Tout est interfacé : 0 échec."
else
  echo "$failures vérification(s) en échec."
  exit 1
fi
