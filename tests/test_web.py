from __future__ import annotations

import asyncio
from collections.abc import Iterator
import json
from pathlib import Path

from fastapi.testclient import TestClient

from cleo1.model_profile import COMPANY_NAME, MODEL_ID, MODEL_NAME, ModelProfile
from cleo1.web import (
    FRONTEND_DIR,
    GenerationRequest,
    _generation_events,
    create_app,
    model_profile_payload,
)


class FakeStoryGenerator:
    summary = "A small fake model · CPU"
    profile = ModelProfile.placeholder()

    def stream_story(
        self,
        prompt: str,
        max_new_tokens: float,
        temperature: float,
        top_k: float,
        seed: float,
    ) -> Iterator[tuple[str, str]]:
        del max_new_tokens, temperature, top_k, seed
        yield prompt, "Starting · 3 prompt tokens"
        yield prompt + " continued.", "Done · 2 tokens · length limit reached"


def test_fastapi_app_exposes_profile_health_and_frontend(tmp_path: Path):
    (tmp_path / "index.html").write_text(
        "<!doctype html><title>Cleo AI — Cleo 1</title><div id='root'></div>",
        encoding="utf-8",
    )
    client = TestClient(create_app(FakeStoryGenerator(), static_dir=tmp_path))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "company": "Cleo AI",
        "model": "Cleo 1",
        "model_id": "cleo-1",
    }

    response = client.get("/api/profile")
    assert response.status_code == 200
    profile = response.json()
    assert profile["identity"]["company_name"] == COMPANY_NAME
    assert profile["identity"]["model_name"] == MODEL_NAME
    assert profile["identity"]["model_id"] == MODEL_ID
    assert profile["identity"]["developed_and_trained_by"] == COMPANY_NAME
    assert profile["identity"]["release"] == "Research release 01"
    assert profile["architecture"]["block_size"] == 256
    assert len(profile["prompt_starters"]) == 5

    page = client.get("/")
    assert page.status_code == 200
    assert "Cleo AI — Cleo 1" in page.text


def test_generation_endpoint_streams_ndjson(tmp_path: Path):
    client = TestClient(create_app(FakeStoryGenerator(), static_dir=tmp_path))
    response = client.post(
        "/api/generate",
        json={
            "prompt": "Once upon a time",
            "max_new_tokens": 32,
            "temperature": 0.8,
            "top_k": 40,
            "seed": 42,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == ["generation", "generation"]
    assert events[-1]["text"] == "Once upon a time continued."
    assert events[-1]["status"].startswith("Done")


def test_identity_endpoint_and_generation_use_canonical_response(tmp_path: Path):
    class CountingGenerator(FakeStoryGenerator):
        calls = 0

        def stream_story(self, *args, **kwargs):
            self.calls += 1
            yield from super().stream_story(*args, **kwargs)

    generator = CountingGenerator()
    client = TestClient(create_app(generator, static_dir=tmp_path))

    identity = client.get("/api/identity")
    assert identity.status_code == 200
    assert identity.json()["company_name"] == "Cleo AI"
    assert identity.json()["model_name"] == "Cleo 1"
    assert identity.json()["model_id"] == "cleo-1"

    response = client.post(
        "/api/generate",
        json={"prompt": "What is your model name and who trained you?"},
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert "Cleo AI" in events[-1]["text"]
    assert "Cleo 1" in events[-1]["text"]
    assert "cleo-1" in events[-1]["text"]
    assert events[-1]["status"].startswith("Done · verified checkpoint identity")
    assert generator.calls == 0


def test_generation_stream_closes_model_iterator_when_client_stops():
    class ClosingStoryGenerator(FakeStoryGenerator):
        closed = False

        def stream_story(self, *args, **kwargs):
            del args, kwargs
            try:
                yield "Once", "Starting"
                yield "Once continued", "Writing"
            finally:
                self.closed = True

    generator = ClosingStoryGenerator()
    request = GenerationRequest(prompt="Once")

    async def stop_after_first_event() -> bytes:
        events = _generation_events(generator, request)
        first = await anext(events)
        await events.aclose()
        return first

    first = asyncio.run(stop_after_first_event())
    assert json.loads(first)["status"] == "Starting"
    assert generator.closed is True


def test_generation_request_rejects_blank_prompt():
    try:
        GenerationRequest(prompt="   ")
    except ValueError as exc:
        assert "Enter a story beginning" in str(exc)
    else:
        raise AssertionError("blank prompt should fail validation")


def test_model_profile_payload_keeps_release_identity():
    profile = ModelProfile.placeholder()
    payload = model_profile_payload(profile)
    assert payload["identity"]["company_name"] == "Cleo AI"
    assert payload["identity"]["model_name"] == "Cleo 1"
    assert payload["identity"]["model_id"] == "cleo-1"
    assert payload["dataset"]["name"] == "roneneldan/TinyStories"
    assert payload["benchmark"]["cached_tokens_per_second"] == 86.5


def test_shadcn_frontend_build_is_packaged():
    index = FRONTEND_DIR / "index.html"
    assert index.is_file()
    html = index.read_text(encoding="utf-8")
    assert '<div id="root"></div>' in html
    assert "Cleo AI — Cleo 1" in html
    assert list((FRONTEND_DIR / "assets").glob("*.js"))
    assert list((FRONTEND_DIR / "assets").glob("*.css"))
