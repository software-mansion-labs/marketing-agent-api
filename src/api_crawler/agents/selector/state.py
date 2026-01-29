from langchain.agents import AgentState

from api_crawler.agents.output_structures import PostChoiceList, PostCritique


class SelectorAgentState(AgentState):
    """Extended state of the Agent."""

    post_critiques: list[PostCritique]
    selection: PostChoiceList
