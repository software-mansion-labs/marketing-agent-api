from typing import Literal

from pydantic import BaseModel, Field

from api_crawler.agents.output_structures import PostHeader
from api_crawler.agents.search import SearchAgentNode


class PostsToLoad(BaseModel):
    """Posts to load."""

    posts: list[PostHeader] = Field(description="list of posts to load")


class LoopDecision(BaseModel):
    """Decision on the next action to take in loop."""

    loop_decision: Literal[SearchAgentNode.SEARCH, SearchAgentNode.SUMMARY] = Field(
        description="whether to keep searching or end the workflow"
    )
