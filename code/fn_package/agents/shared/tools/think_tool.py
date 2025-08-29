from typing import Dict, Any
from .base import Tool
from fn_package.utils.logger import get_logger
from ..thought_store import ThoughtStore

logger = get_logger(__name__)


class ThinkTool(Tool):
    """
    Tool for storing internal thoughts in the agent's ThoughtStore.

    Purpose
    -------
    - Allows the agent to record reasoning steps, context, or decisions.
    - Thoughts are not visible to the user, but guide the agent's internal logic.
    - This tool must never be used in isolation; it should accompany another tool
      that produces an action or message for the user or another agent.
    """

    name = "think"
    definition = {
        "type": "function",
        "name": "think",
        "description": (
            "Create thoughts to store in the agent's internal thought store. "
            "These thoughts are not visible to the user, but help the agent reason "
            "about its actions and decisions. The think tool must never be the only "
            "tool called; always pair it with another tool that produces an action "
            "or message (e.g., hand_over)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "The actual thought content to be stored.",
                },
            },
            "required": ["thought"],
        }
    }

    def __init__(self, thought_store: ThoughtStore):
        """
        Initialize a ThinkTool.

        Parameters
        ----------
        thought_store : ThoughtStore
            The store where generated thoughts will be saved.
        """
        self.thought_store = thought_store

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a thought in the ThoughtStore.

        Parameters
        ----------
        args : Dict[str, Any]
            Arguments containing the "thought" string.

        Returns
        -------
        Dict[str, Any]
            Result dictionary with status and content or error message.
        """
        thought = args.get("thought", "").strip()
        if not thought:
            return {"status": "error", "message": "Thought cannot be empty."}

        try:
            thought = self.thought_store.append(thought)
            return {"status": "success", "content": thought.content}
        except Exception as e:
            logger.exception(f"Failed to create thought: {e}")
            return {"status": "error", "message": str(e)}
