import io
import sys
import os
import uuid
from fn_package.config import DEFAULT_CODE_SESSION_FOLDER
from fn_package.utils.logger import get_logger

logger = get_logger(__name__)

class CodeSession:
    """
    A lightweight Python execution environment that maintains
    its own workspace, global/local namespaces, and execution history.

    Each session:
    - Creates a unique workspace folder.
    - Runs arbitrary Python code with isolated globals/locals.
    - Captures stdout output.
    - Stores a history of executed code snippets.
    """

    def __init__(self, base_dir: str = DEFAULT_CODE_SESSION_FOLDER):
        """
        Initialize a new CodeSession with its own workspace.

        Args:
            base_dir (str): Base directory where the session's workspace will be created.
        """
        # Create a unique workspace directory inside the base_dir
        session_id = uuid.uuid4().hex[:8]
        self._workspace = os.path.abspath(os.path.join(base_dir, f"codesession_{session_id}"))
        os.makedirs(self._workspace, exist_ok=True)

        # Separate execution namespaces
        self._globals = {}
        self._locals = {}
        self._history = []

        logger.info(f"[CodeSession] Initialized with workspace at {self._workspace}")

    def run(self, code: str):
        """
        Execute a string of Python code within the session's workspace.

        - Changes working directory to the session's workspace.
        - Captures stdout output.
        - Restores cwd and stdout after execution.
        - Errors are logged and returned as strings.

        Args:
            code (str): Python code to execute.

        Returns:
            str: Captured stdout or error message.
        """
        self._history.append(code)
        logger.debug(f"[CodeSession] Running code in {self._workspace}:\n{code}")

        buffer = io.StringIO()
        old_stdout = sys.stdout
        old_cwd = os.getcwd()
        sys.stdout = buffer

        try:
            os.chdir(self._workspace)
            exec(code, self._globals, self._locals)
        except Exception as e:
            logger.exception(f"[CodeSession] Error while running code: {e}")
            sys.stdout = old_stdout
            os.chdir(old_cwd)
            return f"Error: {e}"
        finally:
            sys.stdout = old_stdout
            os.chdir(old_cwd)

        output = buffer.getvalue().strip()
        logger.debug(f"[CodeSession] Execution finished. Captured output:\n{output}")
        return output

    def reset(self):
        """
        Reset the session state.

        Clears globals, locals, and history but keeps the workspace directory intact.
        """
        logger.info(f"[CodeSession] Resetting session. Workspace: {self._workspace}")
        self._globals = {}
        self._locals = {}
        self._history = []

    def history(self):
        """
        Get all code snippets executed so far.

        Returns:
            list[str]: Copy of the execution history.
        """
        logger.debug(f"[CodeSession] Returning history. {len(self._history)} items.")
        return list(self._history)

    def workspace(self):
        """
        Get the path to the session's workspace directory.

        Returns:
            str: Absolute path to the workspace.
        """
        logger.debug(f"[CodeSession] Workspace requested: {self._workspace}")
        return self._workspace

    def filetree(self):
        """
        Build and return a textual representation of the workspace file tree.

        Returns:
            str: Multi-line string representing directories and files.
        """
        logger.debug(f"[CodeSession] Building file tree for {self._workspace}")
        tree_lines = []
        for root, dirs, files in os.walk(self._workspace):
            level = root.replace(self._workspace, '').count(os.sep)
            indent = '    ' * level
            tree_lines.append(f"{indent}{os.path.basename(root)}/")
            subindent = '    ' * (level + 1)
            for f in files:
                tree_lines.append(f"{subindent}{f}")
        tree_str = "\n".join(tree_lines)
        logger.debug(f"[CodeSession] File tree:\n{tree_str}")
        return tree_str
