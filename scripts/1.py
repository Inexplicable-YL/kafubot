from dotenv import load_dotenv
from langgraph.store.sqlite import SqliteStore

load_dotenv()


with (
    SqliteStore.from_conn_string(
        "./.database/long_memory.db",
    ) as mstore,
):
    offset = 0
    limit = 10
    all_items = []
    while True:
        items = mstore.search(("long_memory",), offset=offset, limit=limit)
        all_items.extend(items)
        if len(items) < limit:
            break
        offset += limit
    for item in all_items:
        print(item.namespace, item.key, item.value)
