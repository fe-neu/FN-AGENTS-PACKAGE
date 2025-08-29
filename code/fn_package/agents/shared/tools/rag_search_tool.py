from typing import Dict, Any
from .base import Tool
from fn_package.utils.logger import get_logger
from fn_package.retrieval import RagService

logger = get_logger(__name__)


class RagSearchTool(Tool):
    """
    Tool for searching relevant documents via the RAG service.

    Purpose
    -------
    - Retrieves relevant documents from an external knowledge base or document store.
    - Can be configured to use either a fixed number of top-k results or a similarity threshold.
    - Provides structured results for use in reasoning or user responses.

    Notes
    -----
    - Either `k` or `threshold` must be specified.
    - Supplying both will default to using `k`.
    - If neither is specified, the tool call will fail.
    """

    name = "rag_search"
    definition = {
        "type": "function",
        "name": "rag_search",
        "description": (
            "Search for relevant documents using the RAG service to assist in answering user queries. "
            "This tool should be used when the agent needs to retrieve information from an external knowledge base "
            "or document store. You must either define k or a threshold — both or none will result in a crash."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for relevant documents.",
                },
                "k": {
                    "type": "number",
                    "description": "The number of top relevant documents to retrieve.",
                },
                "threshold": {
                    "type": "number",
                    "description": "The similarity threshold for document retrieval.",
                },
            },
            "required": ["query"],
        }
    }

    def __init__(self, rag_service: RagService):
        """
        Initialize the RagSearchTool.

        Parameters
        ----------
        rag_service : RagService
            The retrieval-augmented generation (RAG) service to query against.
        """
        self.rag_service = rag_service

    def run(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a document retrieval query.

        Parameters
        ----------
        args : Dict[str, Any]
            Arguments containing:
            - query (str): The search query (required).
            - k (int, optional): Number of top results to retrieve.
            - threshold (float, optional): Minimum similarity score.

        Returns
        -------
        Dict[str, Any]
            - On success: {"status": "success", "results": [documents]}
            - On failure: {"status": "error", "message": error_reason}
        """
        query = args.get("query", "").strip()
        k = args.get("k", None)
        threshold = args.get("threshold", None)

        # Ensure only one retrieval strategy is used
        if k and threshold:
            threshold = None

        if not query:
            return {"status": "error", "message": "Query cannot be empty."}

        # Perform search
        records, _ = self.rag_service.build_context(query, k=k, threshold=threshold)

        if not records:
            return {"status": "error", "message": "No relevant documents found."}

        # Convert records into plain dicts for response
        docs = []
        for r in records:
            doc = {"id": r.id, "text": r.text}
            if hasattr(r, "source_path"):
                doc["source_path"] = r.source_path
            docs.append(doc)

        return {"status": "success", "results": docs}
