from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from api_crawler.agents import BaseAgent
from api_crawler.agents.critic import CriticAgentNode, CriticAgentState
from api_crawler.agents.output_structures import Critique, Post, PostCritique


class CriticAgent(BaseAgent[CriticAgentState]):
    """AI agent meant to critique suitability of posts for a given task."""

    def __init__(
        self,
        description_prompt: str,
        introduction_prompt: str,
        model: str = "openai:gpt-4o",
    ) -> None:
        """Initializes the Agent's workflow graph and LLM model.

        Args:
            description_prompt (str): description of the product.
            introduction_prompt (str): prompt to use as an introduction of the role of the critic.
            model (str, optional): LLM model to use as foundation for agents. Defaults to "openai:gpt-4o".
        """
        super().__init__(model)
        self._description_prompt = description_prompt
        self._introduction_prompt = introduction_prompt
        self._workflow = self._build_workflow()

    def run(self, posts: list[Post]) -> list[PostCritique]:
        """Runs the Agent.

        Args:
            posts (list[Post]): list of posts to critique.

        Returns:
            list[PostCritique]: critiques.
        """
        responses: list[CriticAgentState] = self._workflow.batch(
            [
                {
                    "post": post.content,
                }
                for post in posts
            ],
            {"recursion_limit": 200},
            return_exceptions=True,
        )

        critiques = [
            PostCritique(post=post.header, critique=response["critique"])
            for post, response in zip(posts, responses)
            if not isinstance(response, Exception)
        ]

        return critiques

    def _build_workflow(
        self,
    ) -> CompiledStateGraph[CriticAgentState, None, CriticAgentState, CriticAgentState]:
        """Builds and compiles the workflow.

        Returns:
            CompiledStateGraph[CriticAgentState, None, CriticAgentState, CriticAgentState]: execution-ready workflow.
        """
        workflow_graph = StateGraph(CriticAgentState)

        workflow_graph.add_node(CriticAgentNode.DESCRIPTION, self._description)
        workflow_graph.add_node(CriticAgentNode.INTRODUCTION, self._introduce)
        workflow_graph.add_node(CriticAgentNode.CRITIQUE, self._criticize)

        workflow_graph.add_edge(START, CriticAgentNode.DESCRIPTION)
        workflow_graph.add_edge(
            CriticAgentNode.DESCRIPTION, CriticAgentNode.INTRODUCTION
        )
        workflow_graph.add_edge(CriticAgentNode.INTRODUCTION, CriticAgentNode.CRITIQUE)
        workflow_graph.add_edge(CriticAgentNode.CRITIQUE, END)

        workflow = workflow_graph.compile()

        return workflow

    def _description(self, _: CriticAgentState) -> CriticAgentState:
        """Introduces the description as context.

        Returns:
            CriticAgentState: update to the state of the Agent
        """
        return {"messages": [SystemMessage(self._description_prompt)]}

    def _introduce(self, _: CriticAgentState) -> CriticAgentState:
        """Introduces the LLM to its task.

        Returns:
            CriticAgentState: update to the state of the Agent.
        """
        return {"messages": [SystemMessage(self._introduction_prompt)]}

    def _criticize(self, state: CriticAgentState) -> CriticAgentState:
        """Critiques the candidate post.

        Args:
            state (CriticAgentState): state of the Agent.

        Returns:
            CriticAgentState: update to the state of the Agent.
        """
        response: Critique = self._invoke_structured_model(
            Critique,
            state["messages"] + [HumanMessage(state["post"])],
        )

        return {"critique": response}
