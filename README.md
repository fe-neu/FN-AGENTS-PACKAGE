# DHBW_ML_private - Code Folder Overview

This repository contains the codebase for a modular multi-agent system designed for document retrieval, memory management, and code execution in Python environments. The `code` folder is the main source directory and includes all core logic, utilities, agents, and supporting files.

## Folder Structure

- `.env`, `.env.template`: Environment variable configuration files.
- `environment.yml`: Conda environment specification.
- `codesessions/`: Contains isolated workspace directories for code execution sessions, each with its own files.
- `data/`: Example data files, such as PDFs and CSVs, used for retrieval and analysis.
- `fn_package/`: **Main Python package** containing all agents, retrieval logic, utilities, and configuration.
- `memory/`: Persistent storage for long-term memory records (CSV).
- `notebooks/`: Example Jupyter notebooks demonstrating usage and testing.
- `scripts/`: Utility scripts for running demos and interacting with the system.

---

## fn_package

The `fn_package` directory is the heart of the system. It is a modular Python package implementing:

### 1. Agents

Located in `fn_package/agents/`, these are specialized classes that interact via a conversation envelope system:

- **HeadAgent**: Coordinates all other agents, makes high-level decisions, and delegates tasks.
- **RagAgent**: Handles document retrieval and synthesis using RAG (Retrieval-Augmented Generation) techniques.
- **MemoryAgent**: Manages long-term memory records about the user, storing and retrieving relevant facts.
- **AnalystAgent**: Executes Python code in a persistent environment for data analysis (CSV, tabular data, plots).
- **MockAgent**: For testing and demonstration purposes.

Agents use prompts (in `prompts/`) to define their behavior and interact via a shared tool system.

### 2. Retrieval

Located in `fn_package/retrieval/`, this module provides:

- **RAG (Retrieval-Augmented Generation)**: 
  - `rag/service.py`: Main service for ingesting documents (PDF/text), chunking, embedding, and retrieval.
  - `rag/parser.py`: PDF parsing using `pypdf`.
  - `rag/chunker.py`: Splits text into manageable chunks for embedding.
  - `core/`: Vector store, embedding, and retrieval logic.
- **Memory**:
  - `memory/service.py`: Manages long-term memory records, retrieval, and context building.
  - `memory/storage.py`: CSV-based persistent storage for memory records.

### 3. Utilities

Located in `fn_package/utils/`, includes:

- `code_session.py`: Manages persistent Python code execution sessions, workspace isolation, and file tree/history tracking.
- `logger.py`: Centralized logging configuration.

### 4. Shared Tools & Thought Store

- `agents/shared/tools/`: Implements tools for agent actions (e.g., running code, searching documents, creating memories, handing over conversations).
- `agents/shared/thought_store.py`: Stores agent "thoughts" for reasoning and planning.

### 5. Configuration

- `config.py`: Loads environment variables and sets system defaults (API keys, model names, chunk sizes, etc.).

---

## Example Workflow

1. **Document Ingestion**: Use [`RagService`](fn_package/retrieval/rag/service.py) to ingest PDFs or text files. Documents are parsed, chunked, embedded, and stored for retrieval.
2. **Conversation Handling**: Agents interact via [`ConversationHandler`](fn_package/conversation/conversation_handler.py), passing envelopes containing messages, sender/recipient info, and timestamps.
3. **Memory Management**: [`MemoryService`](fn_package/retrieval/memory/service.py) stores and retrieves user-relevant facts for personalization.
4. **Code Execution**: [`AnalystAgent`](fn_package/agents/analyst_agent.py) runs Python code in a persistent session using [`CodeSession`](fn_package/utils/code_session.py).
5. **Tool Calls**: Agents use shared tools for specialized actions (e.g., document search, code execution, memory creation).

---

## Notebooks & Scripts

- **notebooks/**: Jupyter notebooks for interactive testing and demonstration.
- **scripts/run_demo.py**: Starts an example chat session using the multi-agent system.

---

## Getting Started

1. Install dependencies using `environment.yml` and activate the environment.
2. Configure `.env`: Copy `.env.template` into the same Directory with your OpenAI API key and other settings.
3. navigate to `code/` and run `python -m scripts.run_demo` to start a demo chat in the Command Line

---

## License & Credits

This codebase is intended for educational and research purposes.