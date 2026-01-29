import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

from api_crawler.agents import CriticAgent, SearchAgent, SelectorAgent
from api_crawler.agents.output_structures import PostChoice
from api_crawler.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class Crawler:
    """Agentic Crawler for searching marketing opportunities on Reddit and Hackernews."""

    def __init__(
        self,
        model: str,
        iterations: int,
        agent_min_iterations: int,
        agent_max_iterations: int,
        description_prompt: str,
        search_search_prompt: str,
        search_select_prompt: str,
        search_decide_loop_prompt: str,
        critic_introduction_prompt: str,
        selector_introduction_prompt: str,
        tags: list[str],
        scrapers: list[BaseScraper],
    ) -> None:
        """Initializes the list of agents.

        Args:
            model (str): ID of the foundation model.
            iterations (int): number of runs for each agent.
            agent_min_iterations (int): min. number of iterations in an agent loop.
            agent_max_iterations (int): max. number of iterations in an agent loop.
            description_prompt (str): prompt with the description of the product.
            search_search_prompt (str): prompt for searching posts.
            search_select_prompt (str): prompt for selecting prompts.
            search_decide_loop_prompt (str): prompt for deciding whether the loop should continue.
            critic_introduction_prompt (str): prompt introducing the role of the critic.
            selector_introduction_prompt (str): prompt introducing the role of the selector.
            tags (list[str]): list of useful tags.
            scrapers (list[BaseScraper]): list of scrapers to use.
        """
        critic = CriticAgent(
            introduction_prompt=critic_introduction_prompt,
            description_prompt=description_prompt,
            model=model,
        )
        selector = SelectorAgent(
            introduction_prompt=selector_introduction_prompt,
            description_prompt=description_prompt,
            model=model,
        )

        self._agents = [
            SearchAgent(
                scraper=scraper,
                tags=tags,
                critic=critic,
                selector=selector,
                search_prompt=search_search_prompt,
                select_prompt=search_select_prompt,
                decide_loop_prompt=search_decide_loop_prompt,
                model=model,
                min_iterations=agent_min_iterations,
                max_iterations=agent_max_iterations,
                description_prompt=description_prompt,
            )
            for scraper in scrapers
        ]

        self._iterations = iterations

    def run(self) -> list[PostChoice]:
        """Runs the crawler and returns found posts.

        Returns:
            list[PostChoice]: found posts.
        """

        def run_agent(
            agent: SearchAgent,
        ) -> list[PostChoice]:
            """Worker thread function.

            Args:
                agent (SearchAgent): single agent to run.

            Returns:
                list[PostChoice]: reply from the agent.
            """
            return agent.run(self._iterations)

        with ThreadPoolExecutor(max_workers=len(self._agents)) as pool:
            results_from_agents = list(pool.map(run_agent, self._agents))

        result_list = list(chain.from_iterable(results_from_agents))

        logger.info(
            f"All agents have completed their runs, found {len(result_list)} posts."
        )

        return result_list
