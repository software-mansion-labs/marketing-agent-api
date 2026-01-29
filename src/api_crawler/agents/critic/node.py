from enum import Enum

from langgraph.graph import END, START


class CriticAgentNode(str, Enum):
    DESCRIPTION = "DESCRIPTION"
    INTRODUCTION = "INTRODUCTION"
    CRITIQUE = "CRITIQUE"
    START = START
    END = END
