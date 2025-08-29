from typing import Dict, Any
from .base import Tool
from fn_package.utils.logger import get_logger
from fn_package.utils import CodeSession

logger = get_logger(__name__)


class ResetCodeSessionTool(Tool):
    """
    Tool for resetting a persistent Python code session.

    Purpose
    -------
    - Clears all variables, imports, and execution history.
    - Preserves the session's workspace directory and any files created within it.
    """

    name = "reset_code_session"
    definition = {
        "type": "function",
        "name": "reset_code_session",
        "description": (
            "Resets the current Python code session. "
            "This clears all variables, imports, and execution history, "
            "but preserves the session's workspace directory and any files created inside it."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        }
    }

    def __init__(self, code_session: CodeSession):
        """
        Initialize the ResetCodeSessionTool.

        Parameters
        ----------
        code_session : CodeSession
            The code session to be reset.
        """
        self.code_session = code_session

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset the current code session.

        Returns
        -------
        Dict[str, Any]
            Result dictionary with reset status and message.
        """
        try:
            logger.info("[ResetCodeSessionTool] Resetting the code session.")
            self.code_session.reset()
            return {
                "status": "success",
                "content": "Code session has been reset. Workspace files remain intact."
            }
        except Exception as e:
            logger.exception(f"[ResetCodeSessionTool] Failed to reset session: {e}")
            return {"status": "error", "message": str(e)}
