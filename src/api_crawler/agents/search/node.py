from enum import Enum

from langgraph.graph import END, START


class SearchAgentNode(str, Enum):
    DESCRIPTION = "DESCRIPTION"
    SEARCH = "SEARCH"
    TOOLS_SEARCHER = "TOOLS_SEARCHER"
    SELECT_POST = "SELECT_POST"
    LOAD = "LOAD"
    CRITIQUE = "CRITIQUE"
    SUMMARY = "SUMMARY"
    START = START
    END = END
