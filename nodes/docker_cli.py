from __future__ import annotations

import contextlib
import shlex
import time
from dataclasses import dataclass
from functools import partial
from typing import Any
from typing_extensions import override

import anyio
from anyio import to_thread
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import PrivateMessageEvent
from sekaibot.exceptions import ParserExit
from sekaibot.internal.rule.utils import ArgumentParser
from sekaibot.permission import User
from sekaibot.rule import Keywords

try:
    import docker
    from docker.errors import APIError, DockerException
except Exception:  # pragma: no cover
    docker = None
    APIError = DockerException = Exception  # type: ignore[assignment,misc]

PROJECT_NAME = "docker"
SERVICES: tuple[str, ...] = (
    "api",
    "worker",
    "worker_beat",
    "web",
    "db_postgres",
    "redis",
    "weaviate",
    "sandbox",
    "plugin_daemon",
    "ssrf_proxy",
    "nginx",
    "init_permissions",
)
SERVICE_ALIASES: dict[str, str] = {
    "db": "db_postgres",
    "postgres": "db_postgres",
    "pg": "db_postgres",
    "workerbeat": "worker_beat",
    "worker-beat": "worker_beat",
    "beat": "worker_beat",
    "plugin": "plugin_daemon",
    "ssrf": "ssrf_proxy",
}
START_ORDER: tuple[str, ...] = (
    "db_postgres",
    "redis",
    "weaviate",
    "sandbox",
    "plugin_daemon",
    "api",
    "worker",
    "worker_beat",
    "web",
    "ssrf_proxy",
    "nginx",
)
STOP_ORDER: tuple[str, ...] = tuple(reversed(START_ORDER))

DEFAULT_LOG_LINES = 120
SERVICE_CALL_TIMEOUT_SECONDS = 30
MAX_REPLY_CHARS = 1500

HELP_TEXT = f"""可用命令：
/docker help
/docker list
/docker status [service|all]
/docker start <service|all>
/docker stop <service|all>
/docker restart <service|all>
/docker logs <service> [--lines N] [--tail N] [--since SEC]

可用 service：
{", ".join(SERVICES)}
"""


@dataclass(slots=True)
class OperationResult:
    ok: bool
    data: Any = None
    error_code: str = ""
    error_message: str = ""

    @classmethod
    def success(cls, data: Any = None) -> OperationResult:
        return cls(ok=True, data=data)

    @classmethod
    def fail(
        cls, error_code: str, error_message: str, data: Any = None
    ) -> OperationResult:
        return cls(
            ok=False,
            data=data,
            error_code=error_code,
            error_message=error_message,
        )


def parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("must be a positive integer")
    return parsed


def build_docker_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="/docker", add_help=False)
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("help")
    subparsers.add_parser("list")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("target", nargs="?", default="all")

    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("target")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("target")

    restart_parser = subparsers.add_parser("restart")
    restart_parser.add_argument("target")

    logs_parser = subparsers.add_parser("logs")
    logs_parser.add_argument("service")
    logs_parser.add_argument(
        "--lines", type=parse_positive_int, default=DEFAULT_LOG_LINES
    )
    logs_parser.add_argument("--tail", type=parse_positive_int, default=None)
    logs_parser.add_argument("--since", type=parse_positive_int, default=None)

    return parser


def chunk_text(text: str, max_chars: int = MAX_REPLY_CHARS) -> list[str]:
    if not text:
        return []
    if max_chars <= 0:
        return [text]

    chunks: list[str] = []
    current = ""

    for line in text.splitlines(keepends=True):
        if len(current) + len(line) <= max_chars:
            current += line
            continue

        if current:
            chunks.append(current.rstrip("\n"))
            current = ""

        while len(line) > max_chars:
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        current = line

    if current:
        chunks.append(current.rstrip("\n"))

    return chunks


def normalize_target(target: str, allow_all: bool) -> str | None:
    value = target.strip().lower().replace("-", "_")
    value = SERVICE_ALIASES.get(value, value)
    if allow_all and value == "all":
        return value
    return value if value in SERVICES else None


def parse_docker_command(text: str) -> Any | ParserExit:
    stripped = text.strip()
    if stripped.startswith("/docker"):
        stripped = stripped[len("/docker") :].strip()

    if not stripped:
        argv = ["help"]
    else:
        try:
            argv = shlex.split(stripped)
        except ValueError as exc:
            return ParserExit(status=2, message=str(exc))

    try:
        return DOCKER_PARSER.parse_args(argv)
    except ParserExit as exc:
        return exc


class DifyDockerService:
    def __init__(self, project_name: str = PROJECT_NAME) -> None:
        self.project_name = project_name

    def list_services(self) -> OperationResult:
        return self._collect_statuses(target="all")

    def get_status(self, target: str) -> OperationResult:
        return self._collect_statuses(target=target)

    def start(self, target: str) -> OperationResult:
        return self._operate("start", target)

    def stop(self, target: str) -> OperationResult:
        return self._operate("stop", target)

    def restart(self, target: str) -> OperationResult:
        if target != "all":
            return self._operate("restart", target)

        stop_result = self._operate("stop", target, ignore_missing=True)
        start_result = self._operate("start", target, ignore_missing=True)
        combined_results = (
            list(stop_result.data.get("results", []))
            + list(start_result.data.get("results", []))
            if stop_result.data and start_result.data
            else []
        )
        ok = stop_result.ok and start_result.ok
        if ok:
            return OperationResult.success(
                {"action": "restart", "results": combined_results}
            )
        return OperationResult.fail(
            "partial_failure",
            "restart all finished with errors",
            {"action": "restart", "results": combined_results},
        )

    def read_logs(
        self,
        service: str,
        *,
        lines: int = DEFAULT_LOG_LINES,
        tail: int | None = None,
        since_seconds: int | None = None,
    ) -> OperationResult:
        def runner(client: Any) -> OperationResult:
            resolved = self._find_container(client, service)
            if not resolved.ok:
                return resolved

            container = resolved.data["container"]
            tail_value = tail if tail is not None else lines
            since_value = (
                max(int(time.time()) - since_seconds, 0)
                if since_seconds is not None
                else None
            )

            logs = container.logs(
                tail=tail_value,
                since=since_value,
                stdout=True,
                stderr=True,
            )
            if isinstance(logs, bytes):
                text = logs.decode("utf-8", errors="replace")
            else:
                text = str(logs)

            return OperationResult.success(
                {
                    "service": service,
                    "container_name": resolved.data["container_name"],
                    "text": text.strip("\n"),
                }
            )

        return self._run_with_client(runner)

    def _collect_statuses(self, target: str) -> OperationResult:
        services = SERVICES if target == "all" else (target,)

        def runner(client: Any) -> OperationResult:
            rows: list[dict[str, Any]] = []
            for service in services:
                resolved = self._find_container(client, service)
                if resolved.error_code == "service_not_found" and target == "all":
                    rows.append(
                        {
                            "service": service,
                            "exists": False,
                            "running": False,
                            "status": "missing",
                            "container_name": "",
                        }
                    )
                    continue
                if not resolved.ok:
                    return resolved

                container = resolved.data["container"]
                with contextlib.suppress(Exception):
                    container.reload()
                status = self._safe_status(container)
                rows.append(
                    {
                        "service": service,
                        "exists": True,
                        "running": status == "running",
                        "status": status,
                        "container_name": resolved.data["container_name"],
                    }
                )
            return OperationResult.success({"services": rows})

        return self._run_with_client(runner)

    def _operate(
        self,
        action: str,
        target: str,
        *,
        ignore_missing: bool = False,
    ) -> OperationResult:
        order = self._ordered_services(action, target)

        def runner(client: Any) -> OperationResult:
            results: list[dict[str, Any]] = []
            success = True
            for service in order:
                op = self._operate_single(
                    client,
                    action=action,
                    service=service,
                    ignore_missing=ignore_missing and target == "all",
                )
                if not op["ok"]:
                    success = False
                results.append(op)

            payload = {"action": action, "results": results}
            if success:
                return OperationResult.success(payload)
            return OperationResult.fail(
                "partial_failure",
                f"{action} finished with errors",
                payload,
            )

        return self._run_with_client(runner)

    def _operate_single(
        self,
        client: Any,
        *,
        action: str,
        service: str,
        ignore_missing: bool,
    ) -> dict[str, Any]:
        resolved = self._find_container(client, service)
        if not resolved.ok:
            if resolved.error_code == "service_not_found" and ignore_missing:
                return {
                    "service": service,
                    "container_name": "",
                    "ok": True,
                    "before": "missing",
                    "after": "missing",
                    "message": "container not found, skipped",
                }
            return {
                "service": service,
                "container_name": "",
                "ok": False,
                "before": "",
                "after": "",
                "message": resolved.error_message,
            }

        container = resolved.data["container"]
        container_name = resolved.data["container_name"]
        before = self._safe_status(container, reload=True)

        try:
            if action == "start":
                if before != "running":
                    container.start()
            elif action == "stop":
                if before == "running":
                    container.stop(timeout=20)
            elif action == "restart":
                container.restart(timeout=20)
            else:
                return {
                    "service": service,
                    "container_name": container_name,
                    "ok": False,
                    "before": before,
                    "after": before,
                    "message": f"unknown action: {action}",
                }
        except APIError as exc:
            return {
                "service": service,
                "container_name": container_name,
                "ok": False,
                "before": before,
                "after": before,
                "message": self._format_api_error(exc),
            }
        except DockerException as exc:
            return {
                "service": service,
                "container_name": container_name,
                "ok": False,
                "before": before,
                "after": before,
                "message": str(exc),
            }

        after = self._safe_status(container, reload=True)
        return {
            "service": service,
            "container_name": container_name,
            "ok": True,
            "before": before,
            "after": after,
            "message": "ok",
        }

    def _ordered_services(self, action: str, target: str) -> tuple[str, ...]:
        if target != "all":
            return (target,)
        if action == "start":
            return START_ORDER
        if action == "stop":
            return STOP_ORDER
        return START_ORDER

    def _find_container(self, client: Any, service: str) -> OperationResult:
        labels = [
            f"com.docker.compose.project={self.project_name}",
            f"com.docker.compose.service={service}",
        ]
        containers = client.containers.list(all=True, filters={"label": labels})

        if not containers:
            # fallback: match by compose service only (useful when project name differs)
            containers = client.containers.list(
                all=True,
                filters={"label": [f"com.docker.compose.service={service}"]},
            )

        if len(containers) > 1:
            names = [self._container_name(container) for container in containers]
            return OperationResult.fail(
                "service_ambiguous",
                f"matched multiple containers: {', '.join(names)}",
            )
        if len(containers) == 1:
            container = containers[0]
            return OperationResult.success(
                {
                    "container": container,
                    "container_name": self._container_name(container),
                }
            )

        fallback_names = (
            f"{self.project_name}-{service}-1",
            f"{service}-1",
        )
        for fallback_name in fallback_names:
            with contextlib.suppress(Exception):
                container = client.containers.get(fallback_name)
                return OperationResult.success(
                    {"container": container, "container_name": fallback_name}
                )

        return OperationResult.fail(
            "service_not_found",
            f"container not found for service '{service}'",
        )

    def _run_with_client(self, runner: Any) -> OperationResult:
        if docker is None:
            return OperationResult.fail(
                "docker_sdk_missing",
                "python docker sdk is not installed",
            )

        client = None
        connect_error: str | None = None
        try:
            client = docker.from_env()
            client.ping()
        except DockerException as exc:
            connect_error = str(exc)
        except Exception as exc:  # pragma: no cover
            connect_error = str(exc)

        if connect_error is not None:
            return OperationResult.fail("docker_unavailable", connect_error)

        try:
            result = runner(client)
        except APIError as exc:
            result = OperationResult.fail(
                "docker_api_error", self._format_api_error(exc)
            )
        except DockerException as exc:
            result = OperationResult.fail("docker_error", str(exc))
        except Exception as exc:  # pragma: no cover
            result = OperationResult.fail("docker_error", str(exc))
        finally:
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close()
        return result

    @staticmethod
    def _format_api_error(exc: Exception) -> str:
        explanation = getattr(exc, "explanation", None)
        if explanation:
            return str(explanation)
        return str(exc)

    @staticmethod
    def _container_name(container: Any) -> str:
        name = getattr(container, "name", "")
        return str(name or "")

    @staticmethod
    def _safe_status(container: Any, *, reload: bool = False) -> str:
        if reload:
            with contextlib.suppress(Exception):
                container.reload()
        status = getattr(container, "status", "unknown")
        return str(status or "unknown")


DOCKER_PARSER = build_docker_parser()


@User("2682064633")
@Keywords("/docker")
class DockerManager(Node[PrivateMessageEvent, dict, Any]):  # type: ignore
    priority = 0
    block = True
    service = DifyDockerService()

    @override
    async def handle(self) -> None:  # noqa: PLR0911
        args = parse_docker_command(self.event.get_plain_text())
        if isinstance(args, ParserExit):
            detail = (args.message or "invalid arguments").strip()
            await self.reply(f"命令参数错误：{detail}\n使用 /docker help 查看帮助。")
            return

        action = str(getattr(args, "action", "help"))

        if action == "help":
            await self.reply(HELP_TEXT)
            return

        if action == "list":
            result = await self._run_service(self.service.list_services)
            await self.reply(self._format_status_result(result, title="Dify 容器列表"))
            return

        if action == "status":
            target = normalize_target(
                str(getattr(args, "target", "all")), allow_all=True
            )
            if target is None:
                await self.reply(self._invalid_service_text())
                return
            result = await self._run_service(self.service.get_status, target)
            await self.reply(
                self._format_status_result(result, title=f"Dify 状态: {target}")
            )
            return

        if action in {"start", "stop", "restart"}:
            target = normalize_target(str(getattr(args, "target", "")), allow_all=True)
            if target is None:
                await self.reply(self._invalid_service_text())
                return
            method = getattr(self.service, action)
            result = await self._run_service(method, target)
            await self.reply(
                self._format_operate_result(result, action=action, target=target)
            )
            return

        if action == "logs":
            service = normalize_target(
                str(getattr(args, "service", "")), allow_all=False
            )
            if service is None:
                await self.reply(self._invalid_service_text(allow_all=False))
                return
            lines = int(getattr(args, "lines", DEFAULT_LOG_LINES))
            tail = getattr(args, "tail", None)
            since = getattr(args, "since", None)
            result = await self._run_service(
                self.service.read_logs,
                service,
                lines=lines,
                tail=tail,
                since_seconds=since,
            )
            await self._reply_logs_result(result)
            return

        await self.reply("未知命令，请使用 /docker help 查看帮助。")

    async def _run_service(self, fn: Any, *args: Any, **kwargs: Any) -> OperationResult:
        try:
            with anyio.fail_after(SERVICE_CALL_TIMEOUT_SECONDS):
                return await to_thread.run_sync(partial(fn, *args, **kwargs))
        except TimeoutError:
            return OperationResult.fail(
                "timeout",
                "docker operation timeout",
            )

    async def _reply_logs_result(self, result: OperationResult) -> None:
        if not result.ok:
            await self.reply(self._error_text(result))
            return

        payload = result.data or {}
        service = str(payload.get("service", ""))
        text = str(payload.get("text", ""))
        if not text:
            await self.reply(f"{service} 最近无日志。")
            return

        chunks = chunk_text(text)
        if len(chunks) == 1:
            await self.reply(f"{service} 日志：\n{chunks[0]}")
            return

        await self.reply(f"{service} 日志共 {len(chunks)} 段：")
        for index, chunk in enumerate(chunks, start=1):
            await self.reply(f"[{index}/{len(chunks)}]\n{chunk}")

    def _format_status_result(self, result: OperationResult, *, title: str) -> str:
        if not result.ok:
            return self._error_text(result)

        services = list((result.data or {}).get("services", []))
        lines = [title]
        for item in services:
            service = item.get("service", "")
            if not item.get("exists", False):
                lines.append(f"{service}: 未部署")
                continue
            running = "运行中" if item.get("running", False) else "已停止"
            name = item.get("container_name", "")
            status = item.get("status", "unknown")
            lines.append(f"{service}: {running} ({name}, status={status})")
        return "\n".join(lines)

    def _format_operate_result(
        self, result: OperationResult, *, action: str, target: str
    ) -> str:
        if not result.data:
            return self._error_text(result)

        payload = result.data
        rows = list(payload.get("results", []))
        title = f"{action} {target}: {'成功' if result.ok else '部分失败'}"
        lines = [title]
        for row in rows:
            service = row.get("service", "")
            container_name = row.get("container_name", "")
            ok = row.get("ok", False)
            before = row.get("before", "")
            after = row.get("after", "")
            message = row.get("message", "")
            status = "OK" if ok else "FAIL"
            lines.append(
                f"{service}: {status} ({container_name}) {before} -> {after}; {message}"
            )
        if (
            not result.ok
            and result.error_code
            and result.error_code != "partial_failure"
        ):
            lines.append(self._error_text(result))
        return "\n".join(lines)

    def _invalid_service_text(self, *, allow_all: bool = True) -> str:
        choices = ", ".join(SERVICES)
        if allow_all:
            choices = f"all, {choices}"
        return f"非法 service。可用值：{choices}"

    @staticmethod
    def _error_text(result: OperationResult) -> str:
        message = result.error_message or "unknown docker error"
        return f"操作失败：[{result.error_code}] {message}"
