from langchain.agents import AgentState

from api_crawler.agents.output_structures import Critique


class CriticAgentState(AgentState):
    """Extended state of the Agent."""

    post: str
    critique: Critique
