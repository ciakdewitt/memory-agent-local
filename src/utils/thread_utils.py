"""
Utility functions for thread management with PostgreSQL support.
"""

import os
import json
from typing import Dict, List, Any, Optional
import uuid

from src.database.db import get_db_connection

# File storage paths - match these across all files
USER_STORAGE = "user_storage.json"
THREAD_STORAGE = "thread_storage.json"
MEMORY_DIR = "./.memory"

def load_user_data() -> dict:
    """Load the user data from storage."""
    if os.path.exists(USER_STORAGE):
        with open(USER_STORAGE, 'r') as f:
            return json.load(f)
    return {"user_id": str(uuid.uuid4())}

def load_thread_id(user_id: str) -> Optional[str]:
    """Load the thread ID from storage."""
    try:
        # Try PostgreSQL first
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM user_threads WHERE user_id = %s",
                (user_id,)
            )
            result = cur.fetchone()
            if result:
                return result[0]
    except Exception as e:
        print(f"Error getting thread ID from PostgreSQL: {str(e)}")
    
    # Fallback to file-based
    if os.path.exists(THREAD_STORAGE):
        with open(THREAD_STORAGE, 'r') as f:
            data = json.load(f)
            return data.get(user_id)
    
    return None

def get_all_threads(user_id: str) -> List[Dict[str, Any]]:
    """Get all threads for a user with their metadata."""
    threads = []
    
    # Ensure memory directory exists
    if not os.path.exists(MEMORY_DIR):
        os.makedirs(MEMORY_DIR, exist_ok=True)
    
    # Check thread_storage first
    if os.path.exists(THREAD_STORAGE):
        with open(THREAD_STORAGE, 'r') as f:
            try:
                data = json.load(f)
                current_thread_id = data.get(user_id)
            except json.JSONDecodeError:
                current_thread_id = None
    else:
        current_thread_id = None
    
    # Get threads from message_vectors
    try:
        from src.database.db import get_db_connection
        conn = get_db_connection()
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT thread_id, 
                       MAX(created_at) as last_active,
                       array_agg(content) as messages
                FROM message_vectors
                WHERE user_id = %s 
                GROUP BY thread_id
                ORDER BY last_active DESC
            """, (user_id,))
            
            db_threads = cur.fetchall()
            
            for thread_record in db_threads:
                thread_id = thread_record[0]
                last_active = thread_record[1]
                messages = thread_record[2]
                
                # Get the last message as preview
                last_message = ""
                if messages and len(messages) > 0:
                    last_message = messages[-1][:50] + "..."
                
                thread_info = {
                    "thread_id": thread_id,
                    "is_current": thread_id == current_thread_id,
                    "messages": [],
                    "last_message": last_message,
                    "created_at": last_active.isoformat() if last_active else ""
                }
                
                threads.append(thread_info)
    except Exception as e:
        print(f"Error getting threads from database: {str(e)}")
    
    # Also scan the memory directory for thread files (legacy)
    for filename in os.listdir(MEMORY_DIR):
        if filename.endswith('.json'):
            thread_id = filename[:-5]  # Remove .json extension
            
            # Skip if we already have this thread from the database
            if any(t["thread_id"] == thread_id for t in threads):
                continue
            
            # Try to get thread metadata
            thread_info = {
                "thread_id": thread_id,
                "is_current": thread_id == current_thread_id,
                "messages": [],
                "last_message": "",
                "created_at": ""
            }
            
            # Try to read the thread file
            try:
                with open(os.path.join(MEMORY_DIR, filename), 'r') as f:
                    data = json.load(f)
                    
                    # Extract messages
                    messages = data.get("messages", [])
                    thread_info["messages"] = messages
                    
                    # Get last message for preview
                    if messages and len(messages) > 0:
                        # Find the last human message
                        human_messages = [m for m in messages if m.get("type") == "HumanMessage"]
                        if human_messages:
                            thread_info["last_message"] = human_messages[-1].get("content", "")[:50] + "..."
                    
                # Add to list
                threads.append(thread_info)
            except Exception as e:
                print(f"Error reading thread file {filename}: {str(e)}")
    
    # Sort threads by current first, then by ID
    threads.sort(key=lambda t: (not t["is_current"], t["thread_id"]))
    
    return threads

def save_user_data(user_data: dict) -> None:
    """Save the user data to storage."""
    with open(USER_STORAGE, 'w') as f:
        json.dump(user_data, f)

def save_thread_id(user_id: str, thread_id: str) -> None:
    """Save the thread ID to storage."""
    try:
        # Try PostgreSQL first
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Upsert the thread ID
            cur.execute("""
                INSERT INTO user_threads (user_id, value) 
                VALUES (%s, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET value = %s
            """, (user_id, thread_id, thread_id))
    except Exception as e:
        print(f"Error saving thread ID to PostgreSQL: {str(e)}")
        
        # Fallback to file-based method
        data = {}
        if os.path.exists(THREAD_STORAGE):
            with open(THREAD_STORAGE, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        
        data[user_id] = thread_id
        
        with open(THREAD_STORAGE, 'w') as f:
            json.dump(data, f)