# In src/database/postgres_saver.py
from langgraph.checkpoint.postgres import PostgresSaver as BaseSaver
from typing import Any, Dict, List, Optional, Union, Tuple
import contextlib

class CustomPostgresSaver(BaseSaver):
    """Custom PostgreSQL saver that disables pipeline mode."""
    
    @contextlib.contextmanager
    def _cursor(self, pipeline=False):
        """Override to disable pipeline mode."""
        # Always force pipeline=False and use a proper context manager
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()