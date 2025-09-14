import multiprocessing as mp
import re
from typing import TypedDict, cast

from sekaibot import Bot

from pipeline import PipelineProcess

bot = Bot(config_file="config.toml")


class SegmentDelay(TypedDict):
    """返回列表中每个元素的类型定义。"""

    output: str
    delay: float


def split_with_delay(
    input: str,  # noqa: A002
    *,
    coefficient: float = 0.2,
) -> list[SegmentDelay]:
    """将输入数据中的 `output` 字段按空格拆分，并为每个子串添加延迟。"""
    if not input or not isinstance(input, str):
        return []
    raw = input.strip()
    if not raw:
        return []

    segments = re.split(r"\s+", raw)

    result: list[SegmentDelay] = []
    for i, segment in enumerate(segments):
        delay = 0.0 if i == 0 else len(segment) * coefficient
        result.append({"output": segment, "delay": delay})

    return result


@bot.bot_run_hook
async def on_bot_run(_bot: Bot) -> None:
    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    _bot.global_state["_pipeline"]["_private"] = PipelineProcess(
        "private_config.toml", ctx, name="private", other_chains=[split_with_delay]
    )
    # _bot.global_state["_pipeline"]["_group"] = PipelineProcess("group_config.toml")


@bot.bot_exit_hook
async def on_bot_exit(_bot: Bot) -> None:
    await cast("PipelineProcess", _bot.global_state["_pipeline"]["_private"]).aclose()
    # await cast("PipelineProcess", _bot.global_state["_pipeline"]["_group"]).aclose()


bot.run()
