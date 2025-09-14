from typing import cast

from sekaibot import Bot

from pipeline import PipelineProcess

bot = Bot(config_file="config.toml")


@Bot.bot_run_hook
async def on_bot_run(_bot: Bot) -> None:
    _bot.global_state["_pipeline"]["_private"] = PipelineProcess("private_config.toml")
    # _bot.global_state["_pipeline"]["_group"] = PipelineProcess("group_config.toml")


@Bot.bot_exit_hook
async def on_bot_exit(_bot: Bot) -> None:
    await cast("PipelineProcess", _bot.global_state["_pipeline"]["_private"]).aclose()
    # await cast("PipelineProcess", _bot.global_state["_pipeline"]["_group"]).aclose()


bot.run()
