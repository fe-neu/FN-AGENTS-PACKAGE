from typing import Dict, Any, List, Type
from .tools import Tool

from fn_package.utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """
    Registry for managing and executing tools.

    Responsibilities
    ----------------
    - Register tools by name.
    - Ensure uniqueness of registered tools.
    - Run tools by name with given arguments.
    - Provide access to registered tool names or all tool instances.
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Tool] = {}
        logger.info("ToolRegistry initialized")

    def register(self, tool: Tool):
        """
        Register a new tool.

        Parameters
        ----------
        tool : Tool
            The tool to register.

        Raises
        ------
        TypeError
            If the object is not an instance of Tool.
        ValueError
            If a tool with the same name is already registered.
        """
        logger.debug(f"Attempting to register tool: {tool.name if hasattr(tool, 'name') else 'unknown'}")
        
        if not isinstance(tool, Tool):
            error_msg = f"Expected Tool, got {type(tool)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        if tool.name in self._tools:
            error_msg = f"Tool '{tool.name}' already registered"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self._tools[tool.name] = tool
        logger.info(f"Successfully registered tool: {tool.name}")

    def run(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a registered tool by name.

        Parameters
        ----------
        name : str
            Name of the tool to execute.
        args : Dict[str, Any]
            Arguments to pass to the tool's run method.

        Returns
        -------
        Dict[str, Any]
            Execution result, containing either data or an error message.
        """
        logger.debug(f"Running tool '{name}' with args: {args}")
        
        tool = self._tools.get(name)
        if not tool:
            error_msg = f"Unknown tool '{name}'"
            logger.warning(error_msg)
            return {"ok": False, "error": error_msg}
        
        try:
            result = tool.run(args)
            logger.info(f"Tool '{name}' executed successfully")
            logger.debug(f"Tool '{name}' result: {result}")
            return {"ok": True, "data": result}
        except Exception as e:
            error_msg = f"Tool '{name}' execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"ok": False, "error": str(e)}

    def names(self) -> List[str]:
        """
        Get a list of registered tool names.

        Returns
        -------
        List[str]
            List of tool names.
        """
        tool_names = list(self._tools.keys())
        logger.debug(f"Available tools: {tool_names}")
        return tool_names
    
    def all(self):
        """
        Get all registered tool instances.

        Returns
        -------
        List[Tool]
            List of tool objects.
        """
        return list(self._tools.values())
