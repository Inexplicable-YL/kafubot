from chat.agent.tools.get_msgs import get_early_messages
from chat.agent.tools.query_image import query_image
from chat.agent.tools.search_song import search_song
from chat.agent.tools.view_msg import view_forward_message

TOOLS = [
    get_early_messages,
    query_image,
    search_song,
    view_forward_message,
]
