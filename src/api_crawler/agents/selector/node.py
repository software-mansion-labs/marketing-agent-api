from enum import Enum

from langgraph.graph import END, START


class SelectorAgentNode(str, Enum):
    DESCRIPTION = "DESCRIPTION"
    INTRODUCTION = "INTRODUCTION"
    SELECTION = "SELECTION"
    START = START
    END = END
