from langchain.agents import AgentState

from api_crawler.agents.output_structures import Post, PostChoiceList, PostCritique
from api_crawler.agents.search.output_structures import PostsToLoad


class SearchAgentState(AgentState):
    """Extended Agent state."""

    id: int
    iteration: int
    posts_to_load: PostsToLoad
    loaded_posts: list[Post]
    post_critiques: list[PostCritique]
    selection: PostChoiceList
