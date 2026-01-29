import datetime
from abc import ABC, abstractmethod

from langchain.tools import BaseTool


class BaseScraper(ABC):
    """Abstract class for web scrapers adjusted to different APIs."""

    def __init__(
        self, timescope: datetime.timedelta = datetime.timedelta(days=1)
    ) -> None:
        """Initializes timescope.

        Args:
            timescope (datetime.timedelta, optional): how old are the posts we wish to see. Defaults to datetime.timedelta(days=1).
        """
        self._timescope = timescope

    @abstractmethod
    def get_searcher(self) -> BaseTool:
        """Creates a tool for searching posts on the website.

        Returns:
            BaseTool: tool for searching posts.
        """
        pass

    @abstractmethod
    def load(self, url: str) -> str:
        """Loads posts found with search.

        Args:
            url (str): url of the post to load.

        Returns:
            str: loaded post.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Returns string representation of the scraper.

        Returns:
            str: string representation of the scraper
        """
        pass

    def _get_timestamp(self) -> int:
        """Returns the timestamp for checking whether the loaded posts are not too old for our timescope.

        Returns:
            int: timestamp.
        """

        return int(
            datetime.datetime.now().timestamp() - self._timescope.total_seconds()
        )
