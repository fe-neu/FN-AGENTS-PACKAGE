from __future__ import annotations
from datetime import datetime

from fn_package.conversation import Envelope, Conversation
from fn_package.agents.base import Agent
from fn_package.utils.logger import get_logger
from fn_package.config import DEFAULT_LLM_MODEL, OPENAI_API_KEY
from .prompts.memory_agent import SYSTEM_PROMPT
from .shared.tools import CreateMemoryTool

logger = get_logger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class MemoryAgent(Agent):
    """
    MemoryAgent: specialized agent for storing user-related information.

    Uses the OpenAI Responses API with a CreateMemoryTool to:
    - Interpret incoming user messages.
    - Extract and persist relevant information into memory.
    """

    def __init__(self, agent_id: str, name: str, model: str = DEFAULT_LLM_MODEL, **kwargs):
        """
        Initialize the MemoryAgent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent.
        name : str
            Display name of the agent.
        model : str, optional
            OpenAI model to use (default from config).

        Raises
        ------
        RuntimeError
            If the OpenAI package is not installed or the API key is missing.
        """
        super().__init__(agent_id=agent_id, name=name, model=model, **kwargs)

        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run `pip install openai`")

        if OPENAI_API_KEY is None:
            raise RuntimeError("OPENAI_API_KEY not set in environment/.env")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"OpenAIAgent initialized with model={self.model}, id={self.id}")

        # Register memory creation tool
        self.tools.register(
            CreateMemoryTool(memory_service=self.memory),
        )

    def handle(self, conversation: Conversation, incoming: Envelope) -> None:
        """
        Handle an incoming message envelope.

        Steps
        -----
        - Build a minimal message context (system + user message).
        - Call the OpenAI Responses API with tools enabled.
        - Execute memory creation tool calls if present.

        Parameters
        ----------
        conversation : Conversation
            The current conversation history (unused directly here).
        incoming : Envelope
            The incoming message envelope.

        Returns
        -------
        None
            This agent modifies memory but does not return an envelope.
        """
        logger.info(f"[{self.id}] received from {incoming.sender}: {incoming.message}")

        # Build minimal context for memory extraction
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": incoming.message})

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=messages,
                tools=[tool.definition for tool in self.tools.all()],
                tool_choice="required",
            )
        except Exception as e:
            logger.exception(f"OpenAI call failed in {self.id}: {e}")
            return  # alternatively, could return an envelope with an error message

        # Handle tool calls (e.g., memory creation)
        tool_calls = self._tool_calls_from_output(resp.output)
        if tool_calls:
            tool_results = self._handle_tool_calls(tool_calls)
            logger.info(f"[{self.id}] executed {len(tool_results)} tool calls.")
