from __future__ import annotations

from types import SimpleNamespace

import pytest

from dssential.narration import NarrationConfig, generate_narration

SUMMARY = {"method": "last_value", "forecast": [{"prediction": 2}], "warnings": [], "limitations": []}


class FakeOpenAI:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.responses = self

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.kwargs = kwargs
        return SimpleNamespace(output_text="Grounded narration")


class FakeAnthropic:
    def __init__(self) -> None:
        self.messages = self

    def create(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Claude narration")])


class FakeGeminiModels:
    def generate_content(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(text="Gemini narration")


class ExplodingOpenAI:
    def __init__(self) -> None:
        self.responses = self

    def create(self, **kwargs: object) -> None:
        raise RuntimeError("provider detail containing secret")


def test_consent_is_required_before_any_client_call() -> None:
    with pytest.raises(ValueError, match="consent"):
        generate_narration(SUMMARY, NarrationConfig("OpenAI", "model", "secret", False), client=FakeOpenAI())


def test_openai_uses_responses_without_storage() -> None:
    client = FakeOpenAI()
    assert (
        generate_narration(SUMMARY, NarrationConfig("OpenAI", "model", "secret", True), client=client)
        == "Grounded narration"
    )
    assert client.kwargs["store"] is False
    assert "secret" not in str(client.kwargs)


def test_anthropic_and_gemini_fake_clients() -> None:
    assert (
        generate_narration(SUMMARY, NarrationConfig("Anthropic", "model", "secret", True), client=FakeAnthropic())
        == "Claude narration"
    )
    gemini = SimpleNamespace(models=FakeGeminiModels())
    assert (
        generate_narration(SUMMARY, NarrationConfig("Gemini", "model", "secret", True), client=gemini)
        == "Gemini narration"
    )


def test_provider_errors_are_sanitized() -> None:
    with pytest.raises(RuntimeError) as captured:
        generate_narration(
            SUMMARY,
            NarrationConfig("OpenAI", "model", "secret", True),
            client=ExplodingOpenAI(),
        )
    assert "secret" not in str(captured.value)
