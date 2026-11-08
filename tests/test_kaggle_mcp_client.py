"""Unit tests for Kaggle MCP client."""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.submission.kaggle_mcp_client import KaggleMCPClient


class MockHTTPResponse:
    def __init__(self, data: str):
        self._data = data.encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_kaggle_mcp_client_init_token():
    client = KaggleMCPClient(token="TEST_TOKEN_123")
    assert client.token == "TEST_TOKEN_123"


def test_kaggle_mcp_client_init_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_MCP_TOKEN", raising=False)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
    with pytest.raises(ValueError, match="Kaggle MCP token not found"):
        KaggleMCPClient(token=None)


def test_call_tool_success():
    client = KaggleMCPClient(token="TEST_TOKEN")
    mock_payload = {
        "result": {
            "content": [{"type": "text", "text": json.dumps({"status": "OK", "user_rank": 1})}],
            "isError": False,
        },
        "id": 1,
        "jsonrpc": "2.0",
    }
    raw_sse = f"event: message\ndata: {json.dumps(mock_payload)}\n"

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(raw_sse)) as mock_urlopen:
        res = client.call_tool("test_tool", {"param": 1})
        assert res["status"] == "OK"
        assert res["user_rank"] == 1
        assert mock_urlopen.called


def test_call_tool_mcp_error():
    client = KaggleMCPClient(token="TEST_TOKEN")
    mock_payload = {
        "error": {"code": -32600, "message": "Invalid request"},
        "id": 1,
        "jsonrpc": "2.0",
    }
    raw_sse = f"event: message\ndata: {json.dumps(mock_payload)}\n"

    with patch("urllib.request.urlopen", return_value=MockHTTPResponse(raw_sse)):
        with pytest.raises(RuntimeError, match="MCP Error"):
            client.call_tool("bad_tool", {})


def test_push_kernel_code_structure():
    client = KaggleMCPClient(token="TEST_TOKEN")
    with patch.object(client, "call_tool", return_value={"version": 11}) as mock_call:
        res = client.push_kernel_code(script_code="print('hello')", kernel_slug="test_slug")
        assert res["version"] == 11
        args = mock_call.call_args[0][1]["request"]
        assert args["slug"] == "test_slug"
        assert args["kernelExecutionType"] == "SaveAndRunAll"
        assert args["kernelType"] == "notebook"
        assert "cells" in args["text"]
