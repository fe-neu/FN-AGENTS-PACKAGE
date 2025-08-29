from __future__ import annotations
from datetime import datetime
import json
from typing import Dict, List, Any

from fn_package.conversation import Envelope, Conversation
from fn_package.agents.base import Agent
from fn_package.utils.logger import get_logger
from fn_package.config import DEFAULT_LLM_MODEL, OPENAI_API_KEY
from .prompts.head_agent import SYSTEM_PROMPT
from .prompts.shared import (
    THOUGHT_INSTRUCTIONS,
    MEMORY_INSTRUCTIONS,
    CONVERSATION_INSTRUCTIONS,
    TEAM_INTRODUCTION,
    build_thought_message,
    build_memory_message,
)
from .shared.tools import HandOverTool, ThinkTool

logger = get_logger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class HeadAgent(Agent):
    """
    HeadAgent: central coordinator agent.

    Uses the OpenAI Responses API to:
    - Manage the flow of conversation between agents and the user.
    - Call tools such as HandOverTool and ThinkTool.
    - Build structured message prompts including system, conversation, thoughts, and memory.
    """

    def __init__(self, agent_id: str, name: str, model: str = DEFAULT_LLM_MODEL, **kwargs):
        """
        Initialize the HeadAgent with its model and tools.

        Parameters
        ----------
        agent_id : str
            Unique identifier for the agent.
        name : str
            Display name of the agent.
        model : str, optional
            OpenAI model name (default from config).
        """
        super().__init__(agent_id=agent_id, name=name, model=model, **kwargs)

        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run `pip install openai`")

        if OPENAI_API_KEY is None:
            raise RuntimeError("OPENAI_API_KEY not set in environment/.env")

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"OpenAIAgent initialized with model={self.model}, id={self.id}")

        # Register available tools
        self.tools.register(HandOverTool())
        self.tools.register(ThinkTool(thought_store=self.thoughts))

    def handle(self, conversation: Conversation, incoming: Envelope) -> None:
        """
        Main handler for processing an incoming envelope.

        Steps
        -----
        - Build initial messages (system, context, thoughts, memory).
        - Call the OpenAI Responses API with tools enabled.
        - Execute all tool calls except hand_over.
        - Repeat until a hand_over call is produced.
        - Return an Envelope forwarding the message to the chosen recipient.

        Parameters
        ----------
        conversation : Conversation
            The full conversation history.
        incoming : Envelope
            The new message envelope to handle.

        Returns
        -------
        Envelope or None
            The handover envelope to another agent or None if failed.
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

            # Step 2: Execute all non-handover tool calls
            if other_calls:
                tool_results = self._handle_tool_calls(other_calls)
                logger.info(f"[{self.id}] executed {len(tool_results)} tool calls.")

                # Step 2b: Rebuild messages including tool results
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
        Construct the full prompt messages for the LLM.

        Includes:
        - System prompt with team and instructions.
        - Full conversation history.
        - Recent thoughts (last 3).
        - Memory context from MemoryService.

        Parameters
        ----------
        conversation : Conversation
            The conversation so far.
        incoming : Envelope
            The incoming user message.

        Returns
        -------
        list[dict]
            Messages formatted for the OpenAI Responses API.
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
        Update the conversation messages after executing tool calls.

        Steps
        -----
        - Remove last thought and memory context.
        - Insert tool call requests and their results.
        - Append updated thoughts and memory back into the messages.

        Parameters
        ----------
        messages : List[dict]
            Current conversation messages.
        tool_calls : list
            Tool call objects to include in messages.
        tool_results : list of dict
            Results returned from executing tool calls.

        Returns
        -------
        list[dict]
            Updated list of messages including tool results.
        """
        messages = messages.copy()

        # Preserve last memory block
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

        # Add updated thoughts
        messages.append({
            "role": "assistant",
            "content": json.dumps(build_thought_message(self.thoughts.tail(3)))
        })

        # Add memory back
        messages.append(memories_for_inquiry)

        logger.debug(messages)
        return messages
