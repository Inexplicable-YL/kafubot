from agent.extensions.query_image import query_image
from agent.extensions.search_song import search_song
from agent.extensions.view_msg import view_forward_message

TOOLS = [
    query_image,
    search_song,
    view_forward_message,
]
