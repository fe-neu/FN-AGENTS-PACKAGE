from typing import Dict, Any
from .base import Tool
from fn_package.utils.logger import get_logger
from fn_package.utils import CodeSession

logger = get_logger(__name__)


class GetCodeHistoryTool(Tool):
    """
    Tool for retrieving the execution history of a code session.

    Purpose
    -------
    - Returns all code snippets that have been executed so far
      in the current persistent Python code session.
    """

    name = "get_code_history"
    definition = {
        "type": "function",
        "name": "get_code_history",
        "description": (
            "Returns the history of all code snippets executed in the current Python code session. "
            "This includes every piece of code that has been run so far."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    def __init__(self, code_session: CodeSession):
        """
        Initialize the GetCodeHistoryTool.

        Parameters
        ----------
        code_session : CodeSession
            The code session whose history will be retrieved.
        """
        self.code_session = code_session

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch the history of executed code snippets.

        Returns
        -------
        Dict[str, Any]
            Result dictionary with execution status and history list,
            or an error message if retrieval fails.
        """
        try:
            logger.info("[GetCodeHistoryTool] Fetching code execution history.")
            history = self.code_session.history()
            return {"status": "success", "content": history}
        except Exception as e:
            logger.exception(f"[GetCodeHistoryTool] Failed to fetch history: {e}")
            return {"status": "error", "message": str(e)}
