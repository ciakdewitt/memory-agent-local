"""
Simple state manager that uses PostgreSQL directly without LangGraph's saver.
"""

import json
import psycopg
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  


class SimpleStateSaver:
    """
    Simple state saver that stores thread state directly in PostgreSQL
    without using LangGraph's mechanism.
    """
    
    def __init__(self, conn):
        """
        Initialize with a database connection.
        """
        self.conn = conn
        self.setup()
    
    def setup(self):
        """Set up the required table."""
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS simple_thread_states (
                    thread_id TEXT PRIMARY KEY,
                    state JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
    
    def save_state(self, thread_id: str, state: Dict[str, Any]) -> None:
        """Save state for a thread."""
        try:
            # Convert state to serializable format
            serializable_state = self._prepare_state_for_storage(state)
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO simple_thread_states (thread_id, state, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (thread_id) 
                    DO UPDATE SET state = %s, updated_at = NOW()
                """, (thread_id, json.dumps(serializable_state), json.dumps(serializable_state)))
                self.conn.commit()
        except Exception as e:
            print(f"Error saving state: {str(e)}")
            self.conn.rollback()

    
    def _prepare_state_for_storage(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert LangGraph state to JSON-serializable format."""
        serializable = {}
        
        # Handle basic fields
        for key, value in state.items():
            if key == "messages":
                # Convert message objects to dictionaries
                messages = []
                for msg in value:
                    if hasattr(msg, "to_dict"):
                        messages.append(msg.to_dict())
                    elif hasattr(msg, "content") and hasattr(msg, "type"):
                        messages.append({
                            "content": msg.content,
                            "type": msg.__class__.__name__
                        })
                    else:
                        # Skip messages we can't convert
                        continue
                serializable[key] = messages
            else:
                # Just include other fields as-is
                serializable[key] = value
        
        return serializable
    
    def load_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Load state for a thread."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT state
                    FROM simple_thread_states
                    WHERE thread_id = %s
                """, (thread_id,))
                result = cur.fetchone()
                if result and result[0]:
                    raw_state = result[0]
                    return self._convert_state_from_storage(raw_state)
        except Exception as e:
            print(f"Error loading state: {str(e)}")
        return None


    def _convert_state_from_storage(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert stored state back to LangGraph format."""
        converted = raw_state.copy()
        
        # Convert messages back to LangChain message objects
        if "messages" in raw_state:
            messages = []
            for msg_data in raw_state["messages"]:
                msg_type = msg_data.get("type")
                msg_content = msg_data.get("content", "")
                
                if msg_type == "HumanMessage":
                    messages.append(HumanMessage(content=msg_content))
                elif msg_type == "AIMessage":
                    messages.append(AIMessage(content=msg_content))
                elif msg_type == "SystemMessage":
                    messages.append(SystemMessage(content=msg_content))
            
            converted["messages"] = messages
        
        return converted