import logging

from dotenv import load_dotenv

import config
from api_crawler.base_scraper import BaseScraper
from api_crawler.crawler import Crawler
from scrapers import SubredditScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Example use of the Crawler"""
    load_dotenv()

    scrapers: list[BaseScraper] = [
        SubredditScraper(subreddit=subreddit, timescope=config.TIMESCOPE)
        for subreddit in config.SUBREDDITS
    ]

    crawler = Crawler(
        config.MODEL,
        config.ITERATIONS,
        config.AGENT_MIN_ITERATIONS,
        config.AGENT_MAX_ITERATIONS,
        config.DESCRIPTION_PROMPT,
        config.SEARCH_SEARCH_PROMPT,
        config.SEARCH_SELECT_PROMPT,
        config.SEARCH_DECIDE_LOOP_PROMPT,
        config.CRITIC_INTRODUCTION_PROMPT,
        config.SELECTOR_INTRODUCTION_PROMPT,
        config.TAGS,
        scrapers,
    )

    results = crawler.run()

    for result in results:
        print(
            f"TITLE: {result.post.title}\nLINK: {result.post.link}\nJUSTIFICATION: {result.justification}\n\n"
        )


if __name__ == "__main__":
    main()
