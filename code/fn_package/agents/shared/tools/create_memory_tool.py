from typing import Dict, Any
from fn_package.retrieval import MemoryService
from .base import Tool
from fn_package.utils.logger import get_logger

logger = get_logger(__name__)


class CreateMemoryTool(Tool):
    """
    Tool for creating a memory record from user input.

    Purpose
    -------
    - Generates a concise memory entry based on user-provided information.
    - Stores the memory in the configured MemoryService for later retrieval.
    - Allows assigning an importance score to guide retrieval weighting.
    """

    name = "create_memory"
    definition = {
        "type": "function",
        "name": "create_memory",
        "description": "Create a concise memory record from user input for future reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A concise summary of the important information to be stored in memory.",
                },
                "importance": {
                    "type": "number",
                    "description": "A float between 0.0 and 1.0 indicating the importance of this memory, where 1.0 is most important.",
                },
            },
            "required": ["summary", "importance"],
        },
    }

    def __init__(self, memory_service: MemoryService):
        """
        Initialize the CreateMemoryTool.

        Parameters
        ----------
        memory_service : MemoryService
            The memory service responsible for storing created memory records.
        """
        self.memory_service = memory_service

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create and store a memory record.

        Parameters
        ----------
        args : Dict[str, Any]
            Arguments containing:
            - summary (str): The text summary of the memory.
            - importance (float): Importance score between 0.0 and 1.0.

        Returns
        -------
        Dict[str, Any]
            - On success: {"status": "success", "record_id": record_id}
            - On failure: {"status": "error", "message": error_reason}
        """
        summary = args.get("summary", "").strip()
        importance = args.get("importance", 0.5)

        if not summary:
            return {"status": "error", "message": "Summary cannot be empty."}

        try:
            record_id = self.memory_service.add(summary, importance)
            return {"status": "success", "record_id": record_id}
        except Exception as e:
            logger.exception(f"Failed to create memory: {e}")
            return {"status": "error", "message": str(e)}
