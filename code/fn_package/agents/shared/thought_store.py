from .thought import Thought
from fn_package.utils.logger import get_logger

logger = get_logger(__name__)


class ThoughtStore:
    """
    Stores and manages an agent's internal thoughts.

    Responsibilities
    ----------------
    - Keep track of a list of `Thought` objects.
    - Automatically trim the store when exceeding a maximum length.
    - Provide access to all or the most recent thoughts.
    - Support clearing all stored thoughts.
    """

    def __init__(self, max_len: int = 100):
        """
        Initialize a ThoughtStore.

        Parameters
        ----------
        max_len : int, optional
            Maximum number of thoughts to retain (default = 100).
        """
        self.max_len = max_len
        self._store: list[Thought] = []
        logger.info(f"ThoughtStore initialized with max_len={max_len}")

    def append(self, content: str) -> Thought:
        """
        Add a new thought to the store.

        If the number of stored thoughts exceeds `max_len`,
        the oldest thoughts are trimmed.

        Parameters
        ----------
        content : str
            The textual content of the thought.

        Returns
        -------
        Thought
            The newly created Thought object.
        """
        thought = Thought(content=content)
        self._store.append(thought)

        # Trim store if necessary
        if len(self._store) > self.max_len:
            trimmed_count = len(self._store) - self.max_len
            self._store = self._store[-self.max_len:]
            logger.debug(f"Trimmed {trimmed_count} thoughts for agent {getattr(self, 'agent_id', 'unknown')}")

        return thought

    def all(self) -> list[Thought]:
        """
        Return all stored thoughts.

        Returns
        -------
        list[Thought]
            Copy of all stored Thought objects.
        """
        return list(self._store)

    def tail(self, k: int = 5) -> list[str]:
        """
        Return the most recent k thoughts as plain strings.

        Parameters
        ----------
        k : int, optional
            Number of recent thoughts to return (default = 5).

        Returns
        -------
        list[str]
            The most recent k thoughts as strings.
        """
        return [t.content for t in self._store[-k:]]

    def clear(self):
        """
        Clear all stored thoughts.

        Logs the number of thoughts that were cleared.
        """
        count = len(self._store)
        self._store = []
        logger.info(f"Cleared {count} thoughts for agent {getattr(self, 'agent_id', 'unknown')}")
