import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar

from langchain.agents import AgentState
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=AgentState)
K = TypeVar("K", bound=BaseModel)


class BaseAgent(ABC, Generic[T]):
    """Base class for Agents."""

    def __init__(self, model: str = "openai:gpt-4o") -> None:
        """Initializes the chat model.

        Args:
            model (str, optional): LLM model to use as foundation for agents. Defaults to "openai:gpt-4o".
        """
        logger.info("Initializing LLM model.")
        self._model = init_chat_model(model)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Runs the agentic workflow."""
        pass

    @abstractmethod
    def _build_workflow(
        self,
    ) -> CompiledStateGraph[T, None, T, T]:
        """Builds and compiles the workflow.

        Returns:
            CompiledStateGraph[T, None, T, T]: execution-ready workflow.
        """
        pass

    def _invoke_structured_model(
        self, schema: Type[K], messages: list[AnyMessage]
    ) -> K:
        """Invokes the LLM forcing it to return a specified type.

        Args:
            schema (Type[K]): type to return.
            messages (list[AnyMessage]): list of messages.

        Returns:
            K: response.
        """
        structured_llm = self._model.with_structured_output(schema)

        response = structured_llm.invoke(messages)

        if isinstance(response, schema):
            return response
        else:
            raise TypeError(f"Unexpected return type: {type(response)}")
