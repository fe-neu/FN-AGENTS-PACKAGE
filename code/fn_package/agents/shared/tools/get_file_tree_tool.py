from typing import Dict, Any
from .base import Tool
from fn_package.utils.logger import get_logger
from fn_package.utils import CodeSession

logger = get_logger(__name__)


class GetFileTreeTool(Tool):
    """
    Tool for retrieving the file tree of the current code session's workspace.

    Purpose
    -------
    - Provides a string representation of all files and directories
      in the session's workspace directory.
    - Useful for inspecting which files have been created or modified
      during the session.
    """

    name = "get_file_tree"
    definition = {
        "type": "function",
        "name": "get_file_tree",
        "description": (
            "Returns a string representation of the file tree in the current session's workspace directory. "
            "Useful for inspecting which files have been created or modified during the session."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }

    def __init__(self, code_session: CodeSession):
        """
        Initialize the GetFileTreeTool.

        Parameters
        ----------
        code_session : CodeSession
            The code session whose workspace file tree will be retrieved.
        """
        self.code_session = code_session

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch the file tree of the current workspace directory.

        Returns
        -------
        Dict[str, Any]
            Result dictionary with execution status and file tree string,
            or an error message if retrieval fails.
        """
        try:
            logger.info("[GetFileTreeTool] Fetching file tree for workspace.")
            tree = self.code_session.filetree()
            return {"status": "success", "content": tree}
        except Exception as e:
            logger.exception(f"[GetFileTreeTool] Failed to fetch file tree: {e}")
            return {"status": "error", "message": str(e)}
