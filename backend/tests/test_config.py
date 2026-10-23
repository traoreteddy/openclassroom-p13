"""Réglages : une variable d'environnement vide vaut « non renseignée »."""

from app.core.config import Settings


def test_empty_environment_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "")
    monkeypatch.setenv("EMBEDDING_DTYPE", "")
    monkeypatch.setenv("EMBEDDING_DEVICE", "")
    monkeypatch.setenv("AGENT_MODEL", "phi4-mini")

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.embedding_dtype == "auto"
    assert settings.embedding_device is None
    assert settings.agent_model == "phi4-mini"
