from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from sekaibot.exceptions import ParserExit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nodes import docker as docker_node  # noqa: E402

LINES_200 = 200
SINCE_30 = 30
CHUNK_LEN_1000 = 1000
TEXT_LEN_3200 = 3200
CHUNKS_4 = 4


class FakeDockerError(Exception):
    pass


class FakeAPIError(FakeDockerError):
    def __init__(self, explanation: str) -> None:
        super().__init__(explanation)
        self.explanation = explanation


class FakeContainer:
    def __init__(
        self,
        name: str,
        status: str = "running",
        logs_data: bytes | str = b"",
        logs_error: Exception | None = None,
        start_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.status = status
        self.logs_data = logs_data
        self.logs_error = logs_error
        self.start_error = start_error

    def reload(self) -> None:
        return None

    def start(self) -> None:
        if self.start_error:
            raise self.start_error
        self.status = "running"

    def stop(self, _timeout: int = 20) -> None:
        self.status = "exited"

    def restart(self, _timeout: int = 20) -> None:
        self.status = "running"

    def logs(self, **_: Any) -> bytes | str:
        if self.logs_error:
            raise self.logs_error
        return self.logs_data


class FakeContainers:
    def __init__(
        self,
        *,
        by_service: dict[str, list[FakeContainer]] | None = None,
        by_name: dict[str, FakeContainer] | None = None,
    ) -> None:
        self.by_service = by_service or {}
        self.by_name = by_name or {}

    def list(  # noqa: A002
        self,
        all: bool = False,  # noqa: A002
        filters: dict[str, Any] | None = None,
    ) -> list[FakeContainer]:
        _ = all
        labels = (filters or {}).get("label", [])
        service = ""
        for label in labels:
            if isinstance(label, str) and label.startswith("com.docker.compose.service="):
                service = label.split("=", 1)[1]
                break
        return list(self.by_service.get(service, []))

    def get(self, name: str) -> FakeContainer:
        if name in self.by_name:
            return self.by_name[name]
        raise FakeDockerError(f"not found: {name}")


class FakeClient:
    def __init__(self, containers: FakeContainers, *, ping_error: Exception | None = None) -> None:
        self.containers = containers
        self.ping_error = ping_error
        self.closed = False

    def ping(self) -> None:
        if self.ping_error:
            raise self.ping_error

    def close(self) -> None:
        self.closed = True


def patch_docker_module(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    class FakeDockerModule:
        @staticmethod
        def from_env() -> FakeClient:
            return client

    monkeypatch.setattr(docker_node, "docker", FakeDockerModule)
    monkeypatch.setattr(docker_node, "DockerException", FakeDockerError)
    monkeypatch.setattr(docker_node, "APIError", FakeAPIError)


def test_parser_success_cases() -> None:
    parser = docker_node.build_docker_parser()

    list_args = parser.parse_args(["list"])
    assert list_args.action == "list"

    status_args = parser.parse_args(["status"])
    assert status_args.action == "status"
    assert status_args.target == "all"

    logs_args = parser.parse_args(
        ["logs", "api", "--lines", str(LINES_200), "--since", str(SINCE_30)]
    )
    assert logs_args.action == "logs"
    assert logs_args.service == "api"
    assert logs_args.lines == LINES_200
    assert logs_args.since == SINCE_30


def test_parser_missing_required_argument() -> None:
    parser = docker_node.build_docker_parser()
    with pytest.raises(ParserExit):
        parser.parse_args(["logs"])


def test_normalize_target() -> None:
    assert docker_node.normalize_target("api", allow_all=False) == "api"
    assert docker_node.normalize_target("ALL", allow_all=True) == "all"
    assert docker_node.normalize_target("unknown", allow_all=True) is None


def test_chunk_text() -> None:
    text = "a" * TEXT_LEN_3200
    chunks = docker_node.chunk_text(text, max_chars=CHUNK_LEN_1000)
    assert len(chunks) == CHUNKS_4
    assert "".join(chunks) == text


def test_docker_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_node, "docker", None)
    service = docker_node.DifyDockerService()
    result = service.list_services()
    assert not result.ok
    assert result.error_code == "docker_sdk_missing"


def test_docker_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(
        FakeContainers(),
        ping_error=FakeDockerError("daemon down"),
    )
    patch_docker_module(monkeypatch, client)
    service = docker_node.DifyDockerService()
    result = service.list_services()
    assert not result.ok
    assert result.error_code == "docker_unavailable"


def test_service_finds_container_by_label(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeContainer("dify-api-1", status="running")
    client = FakeClient(FakeContainers(by_service={"api": [api]}))
    patch_docker_module(monkeypatch, client)

    service = docker_node.DifyDockerService()
    result = service.get_status("api")

    assert result.ok
    item = result.data["services"][0]
    assert item["service"] == "api"
    assert item["exists"] is True
    assert item["running"] is True
    assert item["container_name"] == "dify-api-1"


def test_service_fallback_to_container_name(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeContainer("dify-api-1", status="exited")
    client = FakeClient(
        FakeContainers(
            by_service={},
            by_name={"dify-api-1": api},
        )
    )
    patch_docker_module(monkeypatch, client)

    service = docker_node.DifyDockerService()
    result = service.get_status("api")

    assert result.ok
    item = result.data["services"][0]
    assert item["exists"] is True
    assert item["running"] is False
    assert item["status"] == "exited"


def test_service_ambiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    api1 = FakeContainer("dify-api-1")
    api2 = FakeContainer("dify-api-2")
    client = FakeClient(FakeContainers(by_service={"api": [api1, api2]}))
    patch_docker_module(monkeypatch, client)

    service = docker_node.DifyDockerService()
    result = service.get_status("api")

    assert not result.ok
    assert result.error_code == "service_ambiguous"


def test_logs_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeContainer("dify-api-1", logs_data=b"")
    client = FakeClient(FakeContainers(by_service={"api": [api]}))
    patch_docker_module(monkeypatch, client)

    service = docker_node.DifyDockerService()
    result = service.read_logs("api", lines=20)

    assert result.ok
    assert result.data["text"] == ""


def test_logs_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeContainer(
        "dify-api-1",
        logs_error=FakeAPIError("boom"),
    )
    client = FakeClient(FakeContainers(by_service={"api": [api]}))
    patch_docker_module(monkeypatch, client)

    service = docker_node.DifyDockerService()
    result = service.read_logs("api", lines=20)

    assert not result.ok
    assert result.error_code == "docker_api_error"


def test_start_single_container_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeContainer(
        "dify-api-1",
        status="exited",
        start_error=FakeAPIError("cannot start"),
    )
    client = FakeClient(FakeContainers(by_service={"api": [api]}))
    patch_docker_module(monkeypatch, client)

    service = docker_node.DifyDockerService()
    result = service.start("api")

    assert not result.ok
    assert result.error_code == "partial_failure"
