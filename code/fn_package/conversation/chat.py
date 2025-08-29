from fn_package.utils.logger import get_logger

from .conversation_handler import ConversationHandler
from .envelope import Envelope

from fn_package.retrieval import (
    MemoryService,
    RagService
)

from fn_package.agents import (
    MemoryAgent,
    HeadAgent,
    RagAgent,
    AnalystAgent
)

logger = get_logger(__name__)


def start_example_chat():
    """
    Start an example multi-agent chat session.

    Steps
    -----
    - Initialize services (MemoryService, RagService).
    - Ingest a demo PDF into the RAG service.
    - Create and register agents (MemoryAgent, HeadAgent, RagAgent, AnalystAgent).
    - Start the interactive chat loop.
    """
    logger.info("Chat session starting…")

    handler = ConversationHandler()

    # Initialize services
    memory_service = MemoryService()
    rag_service = RagService()
    rag_service.ingest_pdf(path="data/Attention_Is_All_You_Need_10.pdf")

    # Define and register agents
    agents = [
        MemoryAgent(agent_id="Agent_2_Mem", name="MemoryAgent", memory=memory_service),
        HeadAgent(agent_id="Agent_3_Head", name="HeadAgent", memory=memory_service),
        RagAgent(agent_id="Agent_4_Rag", name="RAGAgent", memory=memory_service, rag_service=rag_service),
        AnalystAgent(agent_id="Agent_5_Analyst", name="AnalystAgent", memory=memory_service)
    ]

    for agent in agents:
        handler.register_agent(agent)
        logger.info(f"Registered agent: id={agent.id}, name={agent.name}")

    # Start interactive chat loop
    start_chat_from_conversation_handler(handler)


def start_chat_from_conversation_handler(handler: ConversationHandler):
    """
    Run the interactive chat loop using the given conversation handler.

    Parameters
    ----------
    handler : ConversationHandler
        The conversation handler managing agents and message routing.
    """
    logger.info("Conversation handler loop started")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            logger.info("User exited chat session")
            print("Exiting chat.")
            break

        logger.debug(f"User input received: {user_input!r}")

        incoming = Envelope(
            sender="User",
            recipient="HeadAgent",
            message=user_input,
            timestamp=None,
        )

        logger.debug(
            f"Dispatching envelope: sender={incoming.sender}, "
            f"recipient={incoming.recipient}, message={incoming.message!r}"
        )

        try:
            response = handler.run(incoming)
        except Exception as e:
            logger.exception(f"Handler failed for user input {user_input!r}: {e}")
            print("An error occurred, see logs for details.")
            continue

        logger.info(f"Agent {response.sender} replied with: {response.message!r}")
        print(f"{response.sender}: {response.message}")
