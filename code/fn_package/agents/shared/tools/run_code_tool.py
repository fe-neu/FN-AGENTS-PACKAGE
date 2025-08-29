from typing import Dict, Any
from .base import Tool
from fn_package.utils.logger import get_logger
from fn_package.utils import CodeSession

logger = get_logger(__name__)


class RunCodeTool(Tool):
    """
    Tool for executing Python code in a persistent execution session.

    Purpose
    -------
    - Allows an agent to run Python code dynamically.
    - Maintains session state across calls (e.g., variables, imports, files).
    - Executes code inside a dedicated workspace directory.
    """

    name = "run_code"
    definition = {
        "type": "function",
        "name": "run_code",
        "description": (
            "Executes Python code inside a persistent execution session. "
            "The session keeps state between calls (variables, imports, files, etc.) "
            "and runs the code in its own dedicated workspace directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute inside the persistent session. "
                        "Standard output from print statements and other console output "
                        "will be captured and returned as a string."
                    ),
                },
            },
            "required": ["code"],
        }
    }

    def __init__(self, code_session: CodeSession):
        """
        Initialize the RunCodeTool.

        Parameters
        ----------
        code_session : CodeSession
            A persistent execution session for running Python code.
        """
        self.code_session = code_session

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the given Python code inside the persistent session.

        Parameters
        ----------
        args : Dict[str, Any]
            Dictionary containing the "code" key with Python code as a string.

        Returns
        -------
        Dict[str, Any]
            Result dictionary with execution status and output or error message.
        """
        code = args.get("code", "").strip()
        if not code:
            return {"status": "error", "message": "Code cannot be empty."}

        try:
            result = self.code_session.run(code=code)
            return {"status": "success", "content": result}
        except Exception as e:
            logger.exception(f"Failed to run code: {e}")
            return {"status": "error", "message": str(e)}
