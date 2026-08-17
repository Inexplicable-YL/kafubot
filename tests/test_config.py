from pathlib import Path

from kafubot.config import load_config


def test_project_config_uses_one_unified_agent_section() -> None:
    config = load_config(Path(__file__).parents[1] / "config.toml")

    assert config.bot.adapter == "cqhttp"
    assert config.cqhttp.url == "/cqhttp/ws"
    assert "group_596488203" in config.agent.proactive_sessions
    assert not hasattr(config, "chat")
