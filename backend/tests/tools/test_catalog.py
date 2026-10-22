"""Règles du catalogue : nommage, descriptions, idempotence, effets."""

from __future__ import annotations

import difflib
import re

import pytest

from app.tools.base import SideEffect

EXPECTED = {
    "inspect_position",
    "get_opening_theory",
    "evaluate_position",
    "search_opening_knowledge",
    "find_opening_videos",
    "start_deep_analysis",
    "get_deep_analysis",
}
MAX_SIMILARITY = 0.85
NAME_RULE = re.compile(r"^(get|search|find|evaluate|inspect|start)_[a-z]+(_[a-z]+)*$")


@pytest.fixture
def registry(make_registry):
    return make_registry()


def test_catalog_content(registry):
    assert set(registry.names) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_naming_rule(name):
    assert NAME_RULE.match(name), f"{name} ne respecte pas verbe_objet"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_description_template(registry, name):
    description = registry.get(name).description

    assert "À utiliser" in description
    assert "Exemple" in description
    assert "Ne pas utiliser" in description
    assert len(description) < 1200  # reste lisible dans le contexte du modèle


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_input_schema_has_an_example_and_forbids_extra_fields(registry, name):
    spec = registry.get(name)
    schema = spec.input_model.model_json_schema()

    assert schema.get("additionalProperties") is False
    assert spec.input_model.model_config.get("json_schema_extra", {}).get("examples")


def test_only_idempotent_tools_are_retried(registry):
    for name in registry.names:
        spec = registry.get(name)
        if spec.policy.max_attempts > 1:
            assert spec.idempotent, f"{name} est réessayé sans être idempotent"


def test_side_effects(registry):
    effects = {name: registry.get(name).side_effect for name in registry.names}

    assert effects["start_deep_analysis"] is SideEffect.CREATES_JOB
    assert effects["get_opening_theory"] is SideEffect.READ_EXTERNAL
    assert effects["inspect_position"] is SideEffect.NONE


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def test_similarity_detects_near_duplicates():
    assert similarity("analyze_position", "analyse_position") > MAX_SIMILARITY


def test_no_two_tool_names_are_near_duplicates(registry):
    """Deux noms trop ressemblants font hésiter le modèle (règle 2.1 du contrat)."""
    names = registry.names
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]

    too_close = [(a, b) for a, b in pairs if similarity(a, b) > MAX_SIMILARITY]

    assert too_close == []
