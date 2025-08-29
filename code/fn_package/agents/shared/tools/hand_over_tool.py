from typing import Dict, Any
from fn_package.retrieval import MemoryService
from .base import Tool
from fn_package.utils.logger import get_logger

logger = get_logger(__name__)


class HandOverTool(Tool):
    """
    Tool for signaling a conversation handover.

    Purpose
    -------
    - Indicates that the current conversation should be handed over
      to another agent or directly to the user.
    - This tool acts as a routing signal and must not be executed directly.
    """

    name = "hand_over"
    definition = {
        "type": "function",
        "name": "hand_over",
        "description": "Signal a handover of the conversation to another agent or to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "The recipient of the handover. Either 'User' or the name of another agent.",
                },
                "message": {
                    "type": "string",
                    "description": "The message content to be handed over to the recipient.",
                },
            },
            "required": ["recipient", "message"],
        },
    }

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        This tool should never be executed directly.

        Raises
        ------
        Exception
            Always raises an exception, as the tool is only intended
            to signal a handover and not perform an action itself.
        """
        raise Exception(
            "HandOverTool is only a signal to hand over the conversation. "
            "It should never be called directly."
        )
