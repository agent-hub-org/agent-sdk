import os
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.testclient import TestClient
from agent_sdk.middleware.infra import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    VerifyInternalKeyMiddleware,
)
from agent_sdk.context import request_id_var, user_id_var


def make_app(*middleware_specs):
    """Build minimal FastAPI app with given middleware (cls, kwargs) tuples."""
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return PlainTextResponse("pong")

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    for cls, kwargs in middleware_specs:
        app.add_middleware(cls, **kwargs)

    return app


class TestRequestIDMiddleware:
    def test_generates_request_id_when_absent(self):
        app = make_app((RequestIDMiddleware, {}))
        with TestClient(app) as client:
            resp = client.get("/ping")
        assert "x-request-id" in resp.headers
        assert len(resp.headers["x-request-id"]) > 0

    def test_propagates_existing_request_id(self):
        app = make_app((RequestIDMiddleware, {}))
        with TestClient(app) as client:
            resp = client.get("/ping", headers={"X-Request-ID": "my-id-123"})
        assert resp.headers["x-request-id"] == "my-id-123"

    def test_sets_request_id_context_var(self):
        captured = {}
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/ctx")
        async def ctx():
            captured["rid"] = request_id_var.get(None)
            return PlainTextResponse("ok")

        with TestClient(app) as client:
            client.get("/ctx", headers={"X-Request-ID": "ctx-test"})

        assert captured["rid"] == "ctx-test"

    def test_sets_user_id_context_var(self):
        captured = {}
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/ctx")
        async def ctx():
            captured["uid"] = user_id_var.get(None)
            return PlainTextResponse("ok")

        with TestClient(app) as client:
            client.get("/ctx", headers={"X-Request-ID": "r1", "X-User-Id": "user-42"})

        assert captured["uid"] == "user-42"


class TestSecurityHeadersMiddleware:
    def test_sets_all_four_security_headers(self):
        app = make_app((SecurityHeadersMiddleware, {}))
        with TestClient(app) as client:
            resp = client.get("/ping")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert resp.headers.get("x-xss-protection") == "1; mode=block"
        assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"


class TestVerifyInternalKeyMiddleware:
    def test_allows_request_when_no_key_configured(self, monkeypatch):
        monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
        app = make_app((VerifyInternalKeyMiddleware, {"public_paths": frozenset({"/health"})}))
        with TestClient(app) as client:
            resp = client.get("/ping")
        assert resp.status_code == 200

    def test_blocks_request_with_wrong_key(self, monkeypatch):
        monkeypatch.setenv("INTERNAL_API_KEY", "secret")
        app = make_app((VerifyInternalKeyMiddleware, {"public_paths": frozenset({"/health"})}))
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/ping", headers={"X-Internal-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_allows_request_with_correct_key(self, monkeypatch):
        monkeypatch.setenv("INTERNAL_API_KEY", "secret")
        app = make_app((VerifyInternalKeyMiddleware, {"public_paths": frozenset({"/health"})}))
        with TestClient(app) as client:
            resp = client.get("/ping", headers={"X-Internal-API-Key": "secret"})
        assert resp.status_code == 200

    def test_public_path_bypasses_key_check(self, monkeypatch):
        monkeypatch.setenv("INTERNAL_API_KEY", "secret")
        app = make_app((VerifyInternalKeyMiddleware, {"public_paths": frozenset({"/health"})}))
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
