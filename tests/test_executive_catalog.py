from __future__ import annotations

from importlib import import_module

from kafubot.cognition.plugins.loader import discover_plugins
from kafubot.config import PluginSettings, load_config

EXPECTED_PLUGINS = {
    "behavior",
    "conversation",
    "environment",
    "expression",
    "interaction",
    "jargon",
    "memory",
    "meme",
    "reply_effects",
    "search_song",
    "social_signals",
    "summarization",
    "view_message",
    "world_model",
}


def test_project_config_enables_current_executive_pipeline() -> None:
    config = load_config("config.toml")
    catalog = discover_plugins(include_entry_points=False)

    assert {definition.name for definition in catalog.definitions} == EXPECTED_PLUGINS
    assert {
        definition.name for definition in catalog.resolve(config.agent.plugins)
    } == (EXPECTED_PLUGINS)


def test_core_loop_is_composed_from_three_cohesive_plugins() -> None:
    definitions = {
        definition.name: definition
        for definition in discover_plugins(include_entry_points=False).definitions
    }

    assert definitions["environment"].requires == ()
    assert definitions["world_model"].requires == ("environment",)
    assert definitions["interaction"].requires == ("environment", "world_model")
    assert {
        "attention",
        "clock",
        "ingress",
        "model",
        "prompt",
        "replyer",
        "self_state",
    }.isdisjoint(definitions)


def test_every_plugin_definition_lives_in_its_discovered_module_or_package() -> None:
    catalog = discover_plugins(include_entry_points=False)

    for definition in catalog.definitions:
        module = import_module(definition.apply.__module__)
        assert module.plugin is definition


def test_every_component_can_be_selected_or_disabled() -> None:
    catalog = discover_plugins(include_entry_points=False)
    definitions = {definition.name: definition for definition in catalog.definitions}

    for target in definitions:
        settings = {name: PluginSettings(enabled=False) for name in EXPECTED_PLUGINS}

        def enable_with_dependencies(
            name: str,
            selected: dict[str, PluginSettings],
        ) -> None:
            selected[name] = PluginSettings(enabled=True)
            for dependency in definitions[name].requires:
                enable_with_dependencies(dependency, selected)

        enable_with_dependencies(target, settings)
        resolved = {definition.name for definition in catalog.resolve(settings)}
        assert target in resolved
        assert resolved == {name for name, value in settings.items() if value.enabled}
