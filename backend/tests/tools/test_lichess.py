"""Outil get_opening_theory : API Lichess simulée avec respx + fixtures versionnées."""

from __future__ import annotations

import httpx
import pytest

from tests.conftest import ITALIAN_FEN, LICHESS_URL, START_FEN, load_fixture

MASTERS = f"{LICHESS_URL}/masters"
LICHESS_DB = f"{LICHESS_URL}/lichess"


@pytest.fixture
def registry(make_registry):
    return make_registry()


async def test_parses_theory_from_masters(registry, respx_mock):
    respx_mock.get(MASTERS).respond(json=load_fixture("lichess/masters_italian.json"))

    result = await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN})

    data = result.data
    assert data["opening"] == {"eco": "C50", "name": "Italian Game"}
    assert data["total_games"] == 12386
    assert data["in_theory"] is True
    assert [m["san"] for m in data["moves"]] == ["Bc5", "Nf6", "Be7", "d6", "g6"]
    first = data["moves"][0]
    assert first == {
        "san": "Bc5",
        "uci": "f8c5",
        "games": 6870,
        "white_pct": 34.1,
        "draw_pct": 41.5,
        "black_pct": 24.4,
        "average_rating": 2451,
    }
    assert data["reference_games"] == []


async def test_sends_fen_auth_and_options(registry, respx_mock):
    route = respx_mock.get(MASTERS).respond(json=load_fixture("lichess/masters_italian.json"))

    await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN, "max_moves": 3})

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer test-token"
    assert request.url.params["fen"] == ITALIAN_FEN
    assert request.url.params["moves"] == "3"
    assert request.url.params["topGames"] == "0"


async def test_max_moves_truncates_the_answer(registry, respx_mock):
    respx_mock.get(MASTERS).respond(json=load_fixture("lichess/masters_italian.json"))

    data = (await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN, "max_moves": 2})).data

    assert len(data["moves"]) == 2


async def test_reference_games(registry, respx_mock):
    route = respx_mock.get(MASTERS).respond(json=load_fixture("lichess/masters_italian.json"))

    data = (
        await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN, "include_games": True})
    ).data

    assert route.calls.last.request.url.params["topGames"] == "4"
    carlsen, caruana = data["reference_games"]
    assert carlsen["white"] == "Carlsen, Magnus"
    assert carlsen["winner"] == "white"
    assert carlsen["url"] == "https://lichess.org/Ys5vKQWj"
    assert caruana["winner"] == "draw"  # winner null = nulle


async def test_lichess_database_uses_rating_and_speed_filters(registry, respx_mock):
    route = respx_mock.get(LICHESS_DB).respond(json=load_fixture("lichess/lichess_rare.json"))

    data = (
        await registry.execute("get_opening_theory", {"fen": START_FEN, "database": "lichess"})
    ).data

    params = route.calls.last.request.url.params
    assert params["speeds"] == "blitz,rapid,classical"
    assert params["ratings"] == "1600,1800,2000,2200,2500"
    assert data["database"] == "lichess"


@pytest.mark.parametrize(
    ("payload", "in_theory"),
    [
        (load_fixture("lichess/lichess_rare.json"), False),  # 12 parties < 20
        ({"white": 0, "draws": 0, "black": 0, "moves": [], "topGames": []}, False),
        (load_fixture("lichess/masters_italian.json"), True),
    ],
    ids=["peu-de-parties", "position-inconnue", "ouverture-classique"],
)
async def test_in_theory_threshold(registry, respx_mock, payload, in_theory):
    respx_mock.get(MASTERS).respond(json=payload)

    data = (await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN})).data

    assert data["in_theory"] is in_theory


async def test_unknown_position_has_no_opening_and_zero_pct(registry, respx_mock):
    respx_mock.get(MASTERS).respond(
        json={"moves": [{"uci": "a2a3", "san": "a3", "white": 0, "draws": 0, "black": 0}]}
    )

    data = (await registry.execute("get_opening_theory", {"fen": START_FEN})).data

    assert data["opening"] is None
    assert data["total_games"] == 0
    assert data["moves"][0]["white_pct"] == 0.0


async def test_no_token_sends_no_auth_header(make_registry, respx_mock):
    route = respx_mock.get(MASTERS).respond(json=load_fixture("lichess/masters_italian.json"))

    await make_registry(lichess_token=None).execute("get_opening_theory", {"fen": ITALIAN_FEN})

    assert "Authorization" not in route.calls.last.request.headers


# --- Erreurs ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "kind", "code", "calls"),
    [
        (httpx.Response(401), "unavailable", "auth_required", 1),
        (httpx.Response(429, headers={"Retry-After": "90"}), "unavailable", "rate_limited", 1),
        (httpx.Response(429), "unavailable", "rate_limited", 1),
        (httpx.Response(400, text="bad fen"), "input", "rejected_by_service", 1),
        (httpx.Response(200, text="<html>"), "unavailable", "bad_response", 1),
        (httpx.Response(503), "unavailable", "retries_exhausted", 3),
    ],
    ids=["401", "429-retry-after", "429-sans-entete", "400", "pas-du-json", "503-persistant"],
)
async def test_http_errors(registry, respx_mock, response, kind, code, calls):
    route = respx_mock.get(MASTERS).mock(return_value=response)

    result = await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN})

    assert result.error["kind"] == kind
    assert result.error["code"] == code
    assert route.call_count == calls  # seul le 503 (transitoire) est réessayé


async def test_rate_limit_tells_how_long_to_wait(registry, respx_mock):
    respx_mock.get(MASTERS).respond(429, headers={"Retry-After": "90"})

    result = await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN})

    assert result.error["details"]["retry_after_s"] == 90
    assert "Attendre 90 s" in result.error["hint"]


@pytest.mark.parametrize(
    ("exception", "code"),
    [
        (httpx.ReadTimeout("lent"), "timeout"),
        (httpx.ConnectError("refusé"), "network_error"),
    ],
)
async def test_network_failures_are_retried_then_succeed(registry, respx_mock, exception, code):
    route = respx_mock.get(MASTERS).mock(
        side_effect=[
            exception,
            httpx.Response(200, json=load_fixture("lichess/masters_italian.json")),
        ]
    )

    result = await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN})

    assert result.ok
    assert route.call_count == 2


async def test_persistent_timeout_reports_last_error(registry, respx_mock):
    respx_mock.get(MASTERS).mock(side_effect=httpx.ReadTimeout("lent"))

    result = await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN})

    assert result.error["code"] == "retries_exhausted"
    assert result.error["details"]["last_error"]["code"] == "timeout"


async def test_breaker_opens_after_repeated_outages(registry, respx_mock):
    route = respx_mock.get(MASTERS).respond(503)

    for _ in range(5):
        await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN})
    result = await registry.execute("get_opening_theory", {"fen": ITALIAN_FEN})

    assert result.error["code"] == "circuit_open"
    assert route.call_count == 15  # 5 appels x 3 tentatives, puis plus rien


async def test_invalid_fen_never_reaches_lichess(registry, respx_mock):
    route = respx_mock.get(MASTERS)

    result = await registry.execute("get_opening_theory", {"fen": "8/8/8/8/8/8/8/8 w - - 0 1"})

    assert result.error["code"] == "invalid_arguments"
    assert not route.called


@pytest.mark.parametrize(
    "arguments",
    [
        {"fen": ITALIAN_FEN, "max_moves": 0},
        {"fen": ITALIAN_FEN, "max_moves": 13},
        {"fen": ITALIAN_FEN, "database": "fide"},
        {"fen": ITALIAN_FEN, "moves": 5},
    ],
    ids=["max_moves-0", "max_moves-13", "base-inconnue", "champ-inconnu"],
)
async def test_out_of_range_arguments(registry, arguments):
    result = await registry.execute("get_opening_theory", arguments)

    assert result.error["code"] == "invalid_arguments"
