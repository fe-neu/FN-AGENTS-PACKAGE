from datetime import datetime
import threading
from typing import Dict, Optional, TYPE_CHECKING

from fn_package.conversation import Conversation, Envelope
from fn_package.utils.logger import get_logger

if TYPE_CHECKING:
    from fn_package.agents import Agent

logger = get_logger(__name__)


class ConversationHandler:
    """
    Handles the flow of messages (envelopes) between agents and the user.

    Responsibilities
    ----------------
    - Register and manage agents participating in a conversation.
    - Route envelopes between agents until a message is returned to the user.
    - Trigger background processing by the MemoryAgent without blocking.
    """

    def __init__(
            self,
            conversation: Optional[Conversation] = None,
            print_status: bool = True,
            ):
        """
        Initialize a ConversationHandler.

        Parameters
        ----------
        conversation : Conversation, optional
            An existing conversation object to use. If None, a new one is created.
        print_status : bool, optional
            Whether to print message routing status to stdout (default = True).
        """
        self.conversation = conversation or Conversation()
        self.agents: Dict[str, 'Agent'] = {}  # Mapping of agent name → agent instance
        self.print_status = print_status

    def register_agent(self, agent: 'Agent'):
        """
        Register an agent so it can receive messages by recipient name.

        Parameters
        ----------
        agent : Agent
            The agent to register.

        Raises
        ------
        TypeError
            If the provided object is not an instance of Agent.
        """
        from fn_package.agents import Agent  # Imported here to avoid circular imports

        if not isinstance(agent, Agent):
            raise TypeError(f"Expected Agent, got {type(agent)}")

        self.agents[agent.name] = agent
        logger.debug(f"Registered agent: {agent.id}")

    def run(self, incoming: Envelope) -> Envelope:
        """
        Main loop: process envelopes until a message is returned to the user.

        Parameters
        ----------
        incoming : Envelope
            The initial envelope to start the conversation flow.

        Returns
        -------
        Envelope
            The final envelope destined for the user.

        Raises
        ------
        ValueError
            If no agent is registered for the target recipient.
        RuntimeError
            If an agent does not return a reply.
        """
        current = incoming
        self.conversation.add(current)

        # --- Trigger MemoryAgent in a background thread (non-blocking) ---
        memory_agent = self.agents.get("MemoryAgent")
        if memory_agent:
            t = threading.Thread(
                target=self._run_memory_agent,
                args=(memory_agent, current),
                daemon=True
            )
            t.start()

        # --- Main message-passing loop ---
        while True:
            target = current.recipient

            if target == "User":
                if self.print_status:
                    print(f"[{current.sender}] → User")
                return current

            agent = self.agents.get(target)
            if not agent:
                logger.error(f"No agent registered for recipient={target}")
                raise ValueError(f"No agent registered for recipient={target}")

            if self.print_status:
                print(f"[{current.sender}] → {target}")

            reply = agent.handle(self.conversation, current)

            if not reply:
                logger.warning(f"Agent {target} returned no reply.")
                raise RuntimeError(f"Agent {target} did not return a reply.")

            self.conversation.add(reply)
            current = reply

    def _run_memory_agent(self, memory_agent: 'Agent', envelope: Envelope):
        """
        Run the MemoryAgent in the background.

        Errors are caught and logged without interrupting the main loop.

        Parameters
        ----------
        memory_agent : Agent
            The MemoryAgent instance.
        envelope : Envelope
            The envelope passed for processing.
        """
        try:
            memory_agent.handle(self.conversation, envelope)
        except Exception as e:
            logger.exception(f"MemoryAgent failed: {e}")
