import datetime
import logging
import os
from functools import cache

from langchain.tools import BaseTool, tool
from praw import Reddit

from api_crawler.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class SubredditScraper(BaseScraper):
    """Scraper class for a specified subreddit."""

    def __init__(
        self,
        subreddit: str,
        post_limit: int = 20,
        max_comments: int = 5,
        timescope: datetime.timedelta = datetime.timedelta(days=1),
    ) -> None:
        """Initializes the Scraper with timescope, subreddit name and post limit.

        Args:
            subreddit (str): name of the subreddit.
            post_limit (int, optional): maximum number of posts we want to see. Defaults to 20.
            max_comments (int, optional): how many top comments we want to load. Defaults to 5.
            timescope (datetime.timedelta, optional): how old are the posts we wish to see. Defaults to datetime.timedelta(days=1).
        """
        super().__init__(timescope)
        self._subreddit = subreddit
        self._max_comments = max_comments
        self._post_limit = post_limit

        logger.info("Initializing Reddit client.")
        self._reddit = Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="marketing agent",
        )

    def get_searcher(self) -> BaseTool:
        """Generates a tool for searching through the specified subreddit.

        Returns:
            BaseTool: resulting tool which allows to search through the subreddit.
        """

        timestamp = super()._get_timestamp()

        @tool(parse_docstring=True)
        @cache
        def search(query: str) -> str:
            """Searches Reddit's r/{subreddit} for posts on the topic.

            Args:
                query (str): query to search on the subreddit.

            Returns:
                str: found posts.
            """
            try:
                sub = self._reddit.subreddit(self._subreddit)
                results = sub.search(
                    query, time_filter="month", sort="new", limit=self._post_limit
                )

                result = []
                for submission in results:
                    if submission.created > timestamp:
                        title = submission.title
                        permalink = f"https://www.reddit.com{submission.permalink}"
                        result.append({"link": permalink, "title": title})

                return str(result) if result else "No results."

            except Exception as e:
                return f"Error: {e}"

        return search

    def load(self, url: str) -> str:
        """Loads a reddit post and some top comments.

        Args:
            url (str): link to the post.

        Returns:
            str: post content.
        """
        try:
            submission = self._reddit.submission(url=url)
            submission.comments.replace_more(limit=0)

            output = [
                f"Title: {submission.title}",
                f"Author: {submission.author}",
                f"Score: {submission.score}",
                f"Link: https://www.reddit.com{submission.permalink}",
            ]

            if submission.selftext:
                output.append(f"\nPost Content:\n{submission.selftext}")

            output.append(f"\nTop {self._max_comments} Comments:")

            for i, comment in enumerate(submission.comments[: self._max_comments], 1):
                body = comment.body.strip().replace("\n", " ")
                output.append(
                    f"{i}. {body} (by {comment.author}, score: {comment.score})"
                )

            return "\n\n".join(output)

        except Exception as e:
            return f"Error loading post: {e}"

    def __str__(self) -> str:
        """Returns string representation of the scraper.

        Returns:
            str: string representation of the scraper
        """
        return f"Reddit r/{self._subreddit}"
