import json
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langgraph.store.sqlite import SqliteStore

load_dotenv()


with SqliteStore.from_conn_string(
    "./.database/long_memory.db",
    index={
        "embed": OpenAIEmbeddings(
            model="text-embedding-3-large",
            base_url=os.getenv("OPENAI_BASE_URL"),
        ),
        "dims": 3072,
    },
) as store:
    """for item in store.search(("jargon",)):
        print(json.dumps(item.value, ensure_ascii=False, indent=2))"""
    offset = 0
    limit = 10
    all_items = []
    while True:
        items = store.search(
            ("jargon",), filter={"count": 4}, offset=offset, limit=limit
        )
        all_items.extend(items)
        if len(items) < limit:
            break
        offset += limit
    for item in all_items:
        print(json.dumps(item.value, ensure_ascii=False, indent=2))
