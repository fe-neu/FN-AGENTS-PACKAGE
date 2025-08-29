from abc import ABC, abstractmethod
import json
from typing import List, Optional, Dict, Any

from .shared import ToolRegistry
from .shared import ThoughtStore

from fn_package.conversation import Conversation
from fn_package.conversation import Envelope
from fn_package.retrieval import MemoryService

from fn_package.config import DEFAULT_LLM_MODEL
from fn_package.utils.logger import get_logger

logger = get_logger(__name__)


class Agent(ABC):
    """
    Abstract base class for all agents.

    Responsibilities
    ----------------
    - Define a consistent interface for agent implementations.
    - Manage common components like tool registry, thoughts, and memory.
    - Provide utility methods for handling tool calls.
    """

    def __init__(
            self,
            agent_id: str,
            name: str,
            model: Optional[str] = DEFAULT_LLM_MODEL,
            tools: Optional[ToolRegistry] = None,
            thoughts: Optional[ThoughtStore] = None,
            memory: Optional[MemoryService] = None,
            **kwargs,
            ):
        """
        Initialize a new agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent.
        name : str
            Human-readable name of the agent.
        model : str, optional
            LLM model used by the agent (default from config).
        tools : ToolRegistry, optional
            Registry of tools available to the agent.
        thoughts : ThoughtStore, optional
            Store for internal reasoning and thought tracking.
        memory : MemoryService, optional
            Memory service for storing and retrieving knowledge.
        """
        self.id: str = agent_id
        self.name: str = name
        self.model: str = model
        self.tools: ToolRegistry = tools or ToolRegistry()
        self.thoughts: ThoughtStore = thoughts or ThoughtStore()
        self.memory: MemoryService = memory

    @abstractmethod
    def handle(self, conversation: Conversation, incoming: Envelope) -> Optional[Envelope]:
        """
        Main entrypoint for all agents.

        Parameters
        ----------
        conversation : Conversation
            The entire conversation history so far.
        incoming : Envelope
            The incoming message envelope to process.

        Returns
        -------
        Optional[Envelope]
            The agent's reply as a new envelope. Must produce exactly one reply.
        """
        ...

    def _handle_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        """
        Execute a list of tool calls using the ToolRegistry.

        Parameters
        ----------
        tool_calls : list
            Tool calls returned from the model output.

        Returns
        -------
        List[Dict[str, Any]]
            List of tool execution results, each represented as a dictionary.
        """
        logger.info(f"Processing {len(tool_calls)} tool calls")

        results = []
        for i, call in enumerate(tool_calls):
            try:
                name = call.name
                raw_args = call.arguments
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

                logger.debug(f"Parsed arguments for {name}: {args}")

                result = self.tools.run(name=name, args=args)

                logger.info(f"Tool '{name}' executed successfully (call_id: {call.id})")
                logger.debug(f"Tool '{name}' result: {result}")

                results.append({
                    "tool_call_id": call.call_id,
                    "content": result
                })

            except Exception as e:
                call_id = getattr(call, "id", None)
                tool_name = getattr(getattr(call, "function", None), "name", "unknown")

                logger.error(f"Tool call failed - Name: {tool_name}, ID: {call_id}, Error: {str(e)}")
                logger.debug("Tool call exception details", exc_info=True)

                results.append({
                    "tool_call_id": call_id,
                    "content": str(e)
                })

        logger.info(f"Tool execution completed: {len(results)} results generated")
        return results

    def _tool_calls_from_output(self, output) -> List[Any]:
        """
        Extract tool calls from a model output.

        Parameters
        ----------
        output : list
            Model output blocks, potentially containing tool calls.

        Returns
        -------
        List[Any]
            List of tool call objects extracted from the output.
        """
        tool_calls = []

        for block in output:
            if block.type == "function_call":   # Direct tool call
                tool_calls.append(block)
            elif block.type == "message":
                # Extract any nested tool calls from message blocks
                if getattr(block, "tool_calls", None):
                    tool_calls.extend(block.tool_calls)

        if not tool_calls:
            logger.warning(f"[{self.id}] no tool calls found.")

        logger.debug(f"Extracted tool calls: {tool_calls}")

        return tool_calls
