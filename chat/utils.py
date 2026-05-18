import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Segment:
    type: str
    data: dict[str, Any] = field(default_factory=dict)  # 修正为 dict


def parse_message(text: str) -> tuple[list[Segment], str]:
    text = text.replace("：", ":").replace("，", ",").strip()
    segments = []
    pattern = re.compile(r"\[MSG:[^\]]*\]")

    def replacer(match: re.Match) -> str:
        block = match.group(0)
        inner = block[5:-1]
        if "," in inner:
            type_part, params_str = inner.split(",", 1)
        else:
            type_part = inner
            params_str = ""
        typ = type_part.strip()
        if not typ:
            seg = None
        data = {}
        if params_str:
            params_with_end = params_str.strip() + ","
            pairs = re.findall(r"([^\s,=]+)\s*=\s*([^,]*?)\s*(?=,)", params_with_end)
            data = {k.strip(): v.strip() for k, v in pairs}
        seg = Segment(type=typ, data=data)
        if seg:
            segments.append(seg)
            return ""
        return block

    clean_text = pattern.sub(replacer, text)
    return segments, clean_text.strip()
