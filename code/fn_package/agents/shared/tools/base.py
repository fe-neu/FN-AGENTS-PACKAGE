from abc import ABC, abstractmethod
from typing import Dict, Any


class Tool(ABC):
    """
    Abstract base class for all tools.

    A tool is a callable unit that can be registered in a ToolRegistry
    and executed by agents. Each tool must define a name, a structured
    definition (e.g., for function calling), and implement the run method.
    """
    name: str
    definition: Dict[str, Any]

    @abstractmethod
    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with the given arguments.

        Parameters
        ----------
        args : Dict[str, Any]
            Input arguments for the tool.

        Returns
        -------
        Dict[str, Any]
            Standardized dictionary result of the tool execution.
        """
        ...
