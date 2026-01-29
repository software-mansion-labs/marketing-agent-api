import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from more_itertools import unique_everseen

from api_crawler.agents.base_agent import BaseAgent
from api_crawler.agents.critic.agent import CriticAgent
from api_crawler.agents.output_structures import Post, PostChoice
from api_crawler.agents.search import SearchAgentNode, SearchAgentState
from api_crawler.agents.search.output_structures import LoopDecision, PostsToLoad
from api_crawler.agents.selector.agent import SelectorAgent
from api_crawler.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent[SearchAgentState]):
    """AI Agent designed for marketing purposes of React Native Executorch."""

    def __init__(
        self,
        scraper: BaseScraper,
        tags: list[str],
        critic: CriticAgent,
        selector: SelectorAgent,
        description_prompt: str,
        search_prompt: str,
        select_prompt: str,
        decide_loop_prompt: str,
        model: str = "openai:gpt-4o",
        min_iterations: int = 2,
        max_iterations: int = 5,
    ) -> None:
        """Initializes the Agent's workflow and LLM model.

        Args:
            scraper (BaseScraper): scraper for site API.
            tags (list[str]): list of important tags.
            critic (CriticAgent): agent that critiques post candidates.
            selector (SelectorAgent): agent that selects best candidates.
            description_prompt (str): description of the product.
            search_prompt (str): prompt used to search for posts.
            select_prompt (str): prompt used to select posts.
            decide_loop_prompt (str): prompt used to decide on loop.
            model (str, optional): LLM model to use as foundation for agents. Defaults to "openai:gpt-4o".
            min_iterations (int, optional): minimum number of iterations. Defaults to 2.
            max_iterations (int, optional): maximum number of iterations. Defaults to 5.
        """
        assert min_iterations <= max_iterations, (
            "min_iterations must be smaller than max_iterations"
        )
        super().__init__(model)
        self._search_tool = scraper.get_searcher()
        self._scraper = scraper
        self._min_iterations = min_iterations
        self._max_iterations = max_iterations
        self._tags = tags
        self._critic = critic
        self._selector = selector
        self._description_prompt = description_prompt
        self._search_prompt = search_prompt
        self._select_prompt = select_prompt
        self._decide_loop_prompt = decide_loop_prompt
        self._workflow = self._build_workflow()

    def _build_workflow(
        self,
    ) -> CompiledStateGraph[SearchAgentState, None, SearchAgentState, SearchAgentState]:
        """Builds and compiles the workflow.

        Returns:
            CompiledStateGraph[SearchAgentState, None, SearchAgentState, SearchAgentState]: execution-ready workflow.
        """
        workflow_graph = StateGraph(SearchAgentState)

        workflow_graph.add_node(SearchAgentNode.DESCRIPTION, self._description)
        workflow_graph.add_node(SearchAgentNode.SEARCH, self._search)
        workflow_graph.add_node(
            SearchAgentNode.TOOLS_SEARCHER, ToolNode(tools=[self._search_tool])
        )
        workflow_graph.add_node(SearchAgentNode.SELECT_POST, self._select_post)
        workflow_graph.add_node(SearchAgentNode.LOAD, self._load)
        workflow_graph.add_node(SearchAgentNode.CRITIQUE, self._critique)
        workflow_graph.add_node(SearchAgentNode.SUMMARY, self._summarize)

        workflow_graph.add_edge(START, SearchAgentNode.DESCRIPTION)
        workflow_graph.add_edge(SearchAgentNode.DESCRIPTION, SearchAgentNode.SEARCH)
        workflow_graph.add_edge(SearchAgentNode.SEARCH, SearchAgentNode.TOOLS_SEARCHER)
        workflow_graph.add_edge(
            SearchAgentNode.TOOLS_SEARCHER, SearchAgentNode.SELECT_POST
        )
        workflow_graph.add_edge(SearchAgentNode.SELECT_POST, SearchAgentNode.LOAD)
        workflow_graph.add_edge(SearchAgentNode.LOAD, SearchAgentNode.CRITIQUE)
        workflow_graph.add_conditional_edges(
            SearchAgentNode.CRITIQUE,
            self._decide_loop,
            {
                SearchAgentNode.SUMMARY: SearchAgentNode.SUMMARY,
                SearchAgentNode.SEARCH: SearchAgentNode.SEARCH,
            },
        )
        workflow_graph.add_edge(SearchAgentNode.SUMMARY, END)

        workflow = workflow_graph.compile()

        return workflow

    def run(self, tries: int = 1) -> list[PostChoice]:
        """Runs the Agent.

        Args:
            tries (int, optional): how many times to run the agent. Defaults to 1.

        Returns:
            list[PostChoice]: suitable posts and justifications for their suitability.
        """

        logger.info(f"Scraping {str(self._scraper)}. Running the Agent.")

        inputs = [
            {
                "id": id,
                "iteration": 0,
                "loaded_posts": [],
                "posts_to_load": PostsToLoad(posts=[]),
                "post_critiques": [],
                "selection": None,
            }
            for id in range(tries)
        ]

        responses: list[SearchAgentState] = self._workflow.batch(
            inputs, {"recursion_limit": 200}, return_exceptions=True
        )
        responses = [
            response for response in responses if not isinstance(response, Exception)
        ]

        logger.info(
            f"Scraping {str(self._scraper)}. Aggregating results from {tries} runs."
        )

        aggregated_result = list(
            unique_everseen(
                (
                    post
                    for response in responses
                    for post in response["selection"].posts
                ),
                key=lambda choice: choice.post.link,
            ),
        )

        return aggregated_result

    def _description(self, _: SearchAgentState) -> SearchAgentState:
        """Introduces the description as context.

        Returns:
            SearchAgentState: update to the state of the Agent
        """
        return {"messages": [SystemMessage(self._description_prompt)]}

    def _search(self, state: SearchAgentState) -> SearchAgentState:
        """Calls the search tool. Start of the search loop.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(
            f"run ID: {state['id']}. Scraping {str(self._scraper)}. Searching for posts."
        )
        prompt = (
            self._search_prompt
            + """ Tags that might come in handy: """
            + str(self._tags)
        )

        response = self._model.bind_tools([self._search_tool]).invoke(
            state["messages"] + [HumanMessage(prompt)],
        )

        return {
            "messages": [HumanMessage(prompt), response],
            "iteration": state["iteration"] + 1,
        }

    def _select_post(self, state: SearchAgentState) -> SearchAgentState:
        """Selects websites to load from search results.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(
            f"run ID: {state['id']}. Scraping {str(self._scraper)}. Selecting pages to visit."
        )

        prompt = self._select_prompt

        response: PostsToLoad = self._invoke_structured_model(
            PostsToLoad,
            state["messages"] + [HumanMessage(prompt)],
        )

        return {
            "messages": [
                HumanMessage(prompt),
                AIMessage(response.model_dump_json()),
            ],
            "posts_to_load": response,
        }

    def _load(self, state: SearchAgentState) -> SearchAgentState:
        """Loads posts' contents.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(
            f"run ID: {state['id']}. Scraping {str(self._scraper)}. Loading posts."
        )

        loaded_posts = [
            Post(header=post, content=self._scraper.load(post.link))
            for post in state["posts_to_load"].posts
        ]

        return {
            "loaded_posts": state["loaded_posts"] + loaded_posts,
            "posts_to_load": [],
        }

    def _critique(self, state: SearchAgentState) -> SearchAgentState:
        """Calls the Critic for each loaded post.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(
            f"run ID: {state['id']}. Scraping {str(self._scraper)}. Critiquing post candidates."
        )
        critiques = self._critic.run(state["loaded_posts"])

        return {
            "messages": [AIMessage(str(critiques))],
            "loaded_posts": [],
            "post_critiques": state["post_critiques"] + critiques,
        }

    def _summarize(self, state: SearchAgentState) -> SearchAgentState:
        """Calls the Selector to pick suitable posts and justify this decision.

        Args:
            state (SearchAgentState): state of the Agent.

        Returns:
            SearchAgentState: update to the state of the Agent.
        """
        logger.info(
            f"run ID: {state['id']}. Scraping {str(self._scraper)}. Picking the best posts."
        )

        response = self._selector.run(state["post_critiques"])

        logger.info(
            f"run ID: {state['id']}. Scraping {str(self._scraper)}. Run ending."
        )

        return {"selection": response}

    def _decide_loop(self, state: SearchAgentState) -> SearchAgentNode:
        """Decides whether to start a new search loop or return results.

        Returns:
            Literal[SearchAgentNode]: decision.
        """

        if state["iteration"] == self._max_iterations:
            return SearchAgentNode.SUMMARY

        if state["iteration"] < self._min_iterations:
            return SearchAgentNode.SEARCH

        prompt = self._decide_loop_prompt

        response: LoopDecision = self._invoke_structured_model(
            LoopDecision,
            state["messages"] + [HumanMessage(prompt)],
        )

        return response.loop_decision
