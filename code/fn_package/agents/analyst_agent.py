from __future__ import annotations
from datetime import datetime
import json
from typing import Dict, List, Any

from fn_package.conversation import Envelope, Conversation
from fn_package.agents.base import Agent
from fn_package.utils.logger import get_logger
from fn_package.utils import CodeSession
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
from .shared.tools import (
    HandOverTool,
    ThinkTool,
    RunCodeTool,
    GetCodeHistoryTool,
    GetFileTreeTool,
    ResetCodeSessionTool
)

logger = get_logger(__name__)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class AnalystAgent(Agent):
    """
    AnalystAgent: specialized agent for code execution and analysis.

    Uses the OpenAI Responses API and provides tools to:
    - Run code in an isolated session.
    - Inspect code history and file tree.
    - Reset the code session when needed.
    - Manage memory and thought context during interactions.
    """

    def __init__(self, agent_id: str, name: str, model: str = DEFAULT_LLM_MODEL, **kwargs):
        """
        Initialize the AnalystAgent.

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

        # Initialize or reuse a code session
        self.code_session = kwargs.get("code_session") or CodeSession()

        # Register available tools
        self.tools.register(HandOverTool())
        self.tools.register(ThinkTool(thought_store=self.thoughts))
        self.tools.register(RunCodeTool(code_session=self.code_session))
        self.tools.register(GetCodeHistoryTool(code_session=self.code_session))
        self.tools.register(GetFileTreeTool(code_session=self.code_session))
        self.tools.register(ResetCodeSessionTool(code_session=self.code_session))

    def handle(self, conversation: Conversation, incoming: Envelope) -> None:
        """
        Handle an incoming message envelope.

        Steps
        -----
        - Build structured messages (system, context, thoughts, memory).
        - Call the OpenAI Responses API with tools enabled.
        - Execute tool calls (except hand_over) iteratively until hand_over is returned.
        - Return an envelope for handover to the next recipient.

        Parameters
        ----------
        conversation : Conversation
            The full conversation history.
        incoming : Envelope
            The incoming message envelope.

        Returns
        -------
        Envelope or None
            The handover envelope, or None if processing fails.
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

            # Step 2: Execute all non-handover tools
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

        # Step 3: Perform the hand_over if available
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
        Construct the prompt messages for the LLM.

        Includes:
        - System prompt with team instructions.
        - Full conversation history.
        - Recent thoughts (last 3).
        - Memory context for the incoming message.

        Parameters
        ----------
        conversation : Conversation
            The current conversation history.
        incoming : Envelope
            The incoming message envelope.

        Returns
        -------
        list[dict]
            Messages formatted for the OpenAI Responses API.
        """
        messages = []

        # System prompt and instructions
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

        # Add conversation history
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
        - Remove last thought and memory entries.
        - Insert tool call requests and their results.
        - Append updated thoughts and the preserved memory back.

        Parameters
        ----------
        messages : List[dict]
            Current list of messages.
        tool_calls : list
            Tool call objects executed.
        tool_results : list of dict
            Results of executed tool calls.

        Returns
        -------
        list[dict]
            Updated list of messages for the next API call.
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

        # Append updated thoughts
        messages.append({
            "role": "assistant",
            "content": json.dumps(build_thought_message(self.thoughts.tail(3)))
        })

        # Append preserved memory
        messages.append(memories_for_inquiry)

        logger.debug(messages)
        return messages
