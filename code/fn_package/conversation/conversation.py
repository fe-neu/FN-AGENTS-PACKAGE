from typing import List, Optional
from .envelope import Envelope
from fn_package.utils.logger import get_logger

logger = get_logger(__name__)


class Conversation:
    """
    Represents a conversation as a sequence of envelopes exchanged
    between agents and the user.

    Provides utilities for:
    - Adding envelopes to history.
    - Retrieving history or the last message.
    - Converting history into OpenAI-style messages.
    - Iterating and pretty-printing the conversation.
    """

    def __init__(self):
        """Initialize an empty conversation."""
        self._envelopes: List[Envelope] = []

    def add(self, envelope: Envelope):
        """
        Add a new envelope to the conversation history.

        Parameters
        ----------
        envelope : Envelope
            The envelope to add.
        """
        self._envelopes.append(envelope)
        logger.info(
            f"Conversation add: [{envelope.sender}] → [{envelope.recipient}] | {envelope.message}"
        )

    def history(self) -> List[Envelope]:
        """
        Return the full conversation history.

        Returns
        -------
        List[Envelope]
            A list of all envelopes in the conversation.
        """
        logger.debug(f"Conversation history requested (len={len(self._envelopes)})")
        return list(self._envelopes)

    def last(self) -> Optional[Envelope]:
        """
        Return the last envelope in the conversation.

        Returns
        -------
        Optional[Envelope]
            The most recent envelope, or None if empty.
        """
        last_env = self._envelopes[-1] if self._envelopes else None
        if last_env:
            logger.debug(
                f"Conversation last: [{last_env.sender}] → [{last_env.recipient}] | {last_env.message}"
            )
        else:
            logger.debug("Conversation last requested but empty.")
        return last_env

    def conversation_as_openai_messages(self) -> List[dict]:
        """
        Convert the conversation into OpenAI-compatible chat messages.

        Returns
        -------
        List[dict]
            A list of messages formatted for OpenAI chat models.
        """
        messages = []
        for env in self._envelopes:
            if env.sender == "User":
                messages.append(
                    {"role": "user", "content": f"TO {env.recipient}:\n{env.message}"}
                )
            else:
                # Treat any non-User sender as assistant
                messages.append(
                    {"role": "assistant", "content": f"FROM {env.sender} TO {env.recipient}:\n {env.message}"}
                )
        return messages

    def __len__(self) -> int:
        """Return the number of envelopes in the conversation."""
        return len(self._envelopes)

    def __iter__(self):
        """Allow iteration over conversation envelopes."""
        return iter(self._envelopes)

    def pretty_print(self):
        """Print the conversation in a human-readable format (for CLI/debugging)."""
        print("\n--- Conversation ---")
        for env in self._envelopes:
            print(f"[{env.timestamp}] {env.sender} → {env.recipient}: {env.message}")
        print("--------------------\n")
