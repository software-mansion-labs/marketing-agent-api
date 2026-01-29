from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from api_crawler.agents import BaseAgent
from api_crawler.agents.output_structures import PostChoiceList, PostCritique
from api_crawler.agents.selector import SelectorAgentNode, SelectorAgentState


class SelectorAgent(BaseAgent[SelectorAgentState]):
    """AI agent meant to pick the best posts for a task based on their critiques."""

    def __init__(
        self,
        description_prompt: str,
        introduction_prompt: str,
        model: str = "openai:gpt-4o",
    ) -> None:
        """Initializes the Agent's workflow graph and LLM model.

        Args:
            description_prompt (str): description of the product.
            introduction_prompt (str): prompt to use as an introduction of the role of the selector.
            model (str, optional): LLM model to use as foundation for agents. Defaults to "openai:gpt-4o".
        """
        super().__init__(model)
        self._description_prompt = description_prompt
        self._introduction_prompt = introduction_prompt
        self._workflow = self._build_workflow()

    def run(self, post_critiques: list[PostCritique]) -> PostChoiceList:
        """Runs the Agent.

        Args:
            post_critiques (list[PostCritique]): list of posts along with critiques of their suitability.

        Returns:
            PostChoiceList: list of picked posts.
        """
        response = self._workflow.invoke(
            {
                "post_critiques": post_critiques,
            },
            {"recursion_limit": 200},
        )["selection"]

        return response

    def _build_workflow(
        self,
    ) -> CompiledStateGraph[
        SelectorAgentState, None, SelectorAgentState, SelectorAgentState
    ]:
        """Builds and compiles the workflow.

        Returns:
            CompiledStateGraph[SelectorAgentState, None, SelectorAgentState, SelectorAgentState]: execution-ready workflow.
        """
        workflow_graph = StateGraph(SelectorAgentState)

        workflow_graph.add_node(SelectorAgentNode.DESCRIPTION, self._description)
        workflow_graph.add_node(SelectorAgentNode.INTRODUCTION, self._introduce)
        workflow_graph.add_node(SelectorAgentNode.SELECTION, self._select)

        workflow_graph.add_edge(START, SelectorAgentNode.DESCRIPTION)
        workflow_graph.add_edge(
            SelectorAgentNode.DESCRIPTION, SelectorAgentNode.INTRODUCTION
        )
        workflow_graph.add_edge(
            SelectorAgentNode.INTRODUCTION, SelectorAgentNode.SELECTION
        )
        workflow_graph.add_edge(SelectorAgentNode.SELECTION, END)

        workflow = workflow_graph.compile()

        return workflow

    def _description(self, _: SelectorAgentState) -> SelectorAgentState:
        """Introduces the description as context.

        Returns:
            SelectorAgentState: update to the state of the Agent.
        """
        return {"messages": [SystemMessage(self._description_prompt)]}

    def _introduce(self, _: SelectorAgentState) -> SelectorAgentState:
        """Introduces the LLM to its task.

        Returns:
            SelectorAgentState: update to the state of the Agent.
        """
        return {"messages": [SystemMessage(self._introduction_prompt)]}

    def _select(self, state: SelectorAgentState) -> SelectorAgentState:
        """Selects the best posts.

        Args:
            state (SelectorAgentState): state of the Agent.

        Returns:
            SelectorAgentState: update to the state of the Agent.
        """
        response: PostChoiceList = self._invoke_structured_model(
            PostChoiceList,
            state["messages"] + [HumanMessage(str(state["post_critiques"]))],
        )

        return {"selection": response}
