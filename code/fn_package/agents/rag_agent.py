from __future__ import annotations
from datetime import datetime
import json
from typing import Any, Dict, List

from fn_package.conversation import Envelope, Conversation
from fn_package.agents.base import Agent
from fn_package.utils.logger import get_logger
from fn_package.config import DEFAULT_LLM_MODEL, OPENAI_API_KEY
from .prompts.rag_agent import SYSTEM_PROMPT
from .prompts.shared import (
    THOUGHT_INSTRUCTIONS,
    MEMORY_INSTRUCTIONS,
    CONVERSATION_INSTRUCTIONS,
    TEAM_INTRODUCTION,
    build_thought_message,
    build_memory_message,
)
from .shared.tools import HandOverTool, ThinkTool, RagSearchTool

logger = get_logger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class RagAgent(Agent):
    """
    RagAgent: specialized agent for retrieval-augmented generation.

    Uses the OpenAI Responses API to:
    - Retrieve relevant information via RagSearchTool.
    - Manage thoughts, memory, and conversation context.
    - Hand over results to other agents when appropriate.
    """

    def __init__(self, agent_id: str, name: str, model: str = DEFAULT_LLM_MODEL, **kwargs):
        """
        Initialize the RagAgent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent.
        name : str
            Display name of the agent.
        model : str, optional
            OpenAI model name (default from config).

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
        
        # Attach RAG service if provided
        self.rag_service = kwargs.get("rag_service")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"OpenAIAgent initialized with model={self.model}, id={self.id}")

        # Register available tools
        self.tools.register(HandOverTool())
        self.tools.register(ThinkTool(thought_store=self.thoughts))
        self.tools.register(RagSearchTool(rag_service=self.rag_service))

    def handle(self, conversation: Conversation, incoming: Envelope) -> None:
        """
        Handle an incoming message envelope.

        Steps
        -----
        - Build structured messages with system, conversation, thoughts, and memory.
        - Call the OpenAI Responses API with tools enabled.
        - Execute all tool calls except hand_over until a hand_over call is produced.
        - Return an envelope that hands over to the intended recipient.

        Parameters
        ----------
        conversation : Conversation
            The full conversation history.
        incoming : Envelope
            The incoming message envelope.

        Returns
        -------
        Envelope or None
            The envelope for handover, or None if processing fails.
        """
        logger.info(f"[{self.id}] received from {incoming.sender}: {incoming.message}")

        messages = self._build_messages(conversation, incoming)

        try:
            resp = self.client.responses.create(
                model=self.model,
                input=messages,
                tools=[tool.definition for tool in self.tools.all()],
                tool_choice="required",
            )
        except Exception as e:
            logger.exception(f"OpenAI call failed in {self.id}: {e}")
            return
        
        logger.debug(f"[{self.id}] full response: {resp}")

        tool_calls = self._tool_calls_from_output(resp.output)

        logger.debug(f"[{self.id}] tool calls: {len(tool_calls)}")

        handover_call = None
        while not handover_call:
            # Step 1: Separate hand_over from other tool calls
            other_calls = []
            for call in tool_calls:
                if call.name == "hand_over":
                    handover_call = call
                else:
                    other_calls.append(call)

            # Step 2: Execute non-handover tools
            if other_calls:
                tool_results = self._handle_tool_calls(other_calls)
                logger.info(f"[{self.id}] executed {len(tool_results)} tool calls.")

                # Step 2b: Rebuild messages with tool results
                messages = self._build_messages_after_tool_calls(
                    messages=messages, tool_calls=other_calls, tool_results=tool_results
                )

                try:
                    resp = self.client.responses.create(
                        model=self.model,
                        input=messages,
                        tools=[tool.definition for tool in self.tools.all()],
                        tool_choice="required",
                    )
                except Exception as e:
                    logger.exception(f"OpenAI call failed in {self.id}: {e}")
                    return
                
                logger.debug(f"[{self.id}] full response: {resp}")

                tool_calls = self._tool_calls_from_output(resp.output)

                logger.debug(f"[{self.id}] tool calls: {len(tool_calls)}")

        # Step 3: Perform the hand_over (if present)
        if handover_call:
            raw_args = handover_call.arguments
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            recipient = args["recipient"]
            message = args["message"]

            logger.info(f"[{self.id}] handing over to {recipient} with message: {message[:60]}...")
            return Envelope(
                sender=self.name,
                recipient=recipient,
                timestamp=datetime.now(),
                message=message,
            )

        return None

    def _build_messages(self, conversation: Conversation, incoming: Envelope) -> list[dict]:
        """
        Build structured messages for the OpenAI Responses API.

        Includes:
        - System prompt with team instructions.
        - Conversation history.
        - Recent thoughts (last 3).
        - Memory context for the incoming message.

        Parameters
        ----------
        conversation : Conversation
            The conversation history.
        incoming : Envelope
            The incoming message envelope.

        Returns
        -------
        list[dict]
            Messages formatted for the OpenAI API.
        """
        messages = []

        # System prompt and context
        messages.append({
            "role": "system",
            "content": f"""
                {SYSTEM_PROMPT}
                #######
                {TEAM_INTRODUCTION}
                #######
                {CONVERSATION_INSTRUCTIONS}
                #######
                {THOUGHT_INSTRUCTIONS}
                #######
                {MEMORY_INSTRUCTIONS}
            """
        })

        # Conversation history
        messages.extend(conversation.conversation_as_openai_messages())

        # Thought context
        messages.append({
            "role": "assistant",
            "content": json.dumps(build_thought_message(self.thoughts.tail(3)))
        })

        # Memory context
        records, ctx = self.memory.build_context(query=incoming.message, k=3)
        messages.append({
            "role": "assistant",
            "content": json.dumps(build_memory_message(records))
        })

        logger.debug(messages)
        return messages
    
    def _build_messages_after_tool_calls(
        self,
        messages: List[dict],
        tool_calls,
        tool_results: list[Dict[str, Any]]
    ) -> list[dict]:
        """
        Update messages after executing tool calls.

        Steps
        -----
        - Preserve the last memory block.
        - Remove last thoughts and memory entries.
        - Insert tool calls and their results.
        - Append updated thoughts and the preserved memory back.

        Parameters
        ----------
        messages : List[dict]
            Current conversation messages.
        tool_calls : list
            Tool call objects to include in messages.
        tool_results : list of dict
            Results from executing the tool calls.

        Returns
        -------
        list[dict]
            Updated messages with tool results included.
        """
        messages = messages.copy()

        # Preserve the last memory block
        memories_for_inquiry = messages[-1]

        # Remove last thoughts and memory
        messages = messages[:-2]

        # Insert tool calls
        for call in tool_calls:
            messages.append(call)

        # Insert tool results
        for result in tool_results:
            messages.append({
                "type": "function_call_output",
                "call_id": result["tool_call_id"],
                "output": json.dumps(result["content"]),
            })

        # Append updated thoughts
        messages.append({
            "role": "assistant",
            "content": json.dumps(build_thought_message(self.thoughts.tail(3)))
        })

        # Append preserved memory
        messages.append(memories_for_inquiry)

        logger.debug(messages)
        return messages
