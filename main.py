"""
Main application entry point for the Memory Agent demo.
"""

import os
import argparse
from typing import Optional, List, Dict, Any
import json
import uuid

from src.agent.agent import run_agent
from src.database.db import get_db_connection

# File to store user information
USER_STORAGE = "user_storage.json"
THREAD_STORAGE = "thread_storage.json"
MEMORY_DIR = "./.memory"

def load_user_data() -> dict:
    """Load the user data from storage if it exists."""
    if os.path.exists(USER_STORAGE):
        with open(USER_STORAGE, 'r') as f:
            return json.load(f)
    return {"user_id": str(uuid.uuid4())}

def load_thread_id(user_id: str) -> Optional[str]:
    """Load the thread ID from the database."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT value FROM user_threads 
                WHERE user_id = %s
            """, (user_id,))
            result = cur.fetchone()
            if result:
                return result[0]
    except Exception as e:
        print(f"Error loading thread ID: {str(e)}")
        # Fall back to file-based method if database fails
        if os.path.exists(THREAD_STORAGE):
            with open(THREAD_STORAGE, 'r') as f:
                data = json.load(f)
                return data.get(user_id)
    return None

def save_thread_id(user_id: str, thread_id: str) -> None:
    """Save the thread ID to the database."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_threads (user_id, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (user_id) 
                DO UPDATE SET value = %s, updated_at = NOW()
            """, (user_id, thread_id, thread_id))
            conn.commit()
    except Exception as e:
        print(f"Error saving thread ID: {str(e)}")
        # Fall back to file-based method if database fails
        data = {}
        if os.path.exists(THREAD_STORAGE):
            with open(THREAD_STORAGE, 'r') as f:
                data = json.load(f)
        data[user_id] = thread_id
        with open(THREAD_STORAGE, 'w') as f:
            json.dump(data, f)

def list_and_select_threads(user_id: str) -> Optional[str]:
    """
    List all available threads and let the user select one.
    
    Args:
        user_id: The user ID to list threads for
        
    Returns:
        Selected thread ID or None for a new thread
    """
    # Connect to the database
    try:
        conn = get_db_connection()
        
        with conn.cursor() as cur:
            # Get all threads for this user
            cur.execute("""
                SELECT DISTINCT thread_id, 
                       MIN(created_at) as first_message, 
                       MAX(created_at) as last_message,
                       COUNT(*) as message_count,
                       string_agg(CASE WHEN role = 'user' THEN substring(content, 1, 50) ELSE '' END, ' | ' 
                           ORDER BY created_at DESC) as recent_messages
                FROM message_vectors
                WHERE user_id = %s
                GROUP BY thread_id
                ORDER BY last_message DESC
            """, (user_id,))
            
            threads = []
            for row in cur.fetchall():
                thread_id = row[0]
                first_message = row[1]
                last_message = row[2]
                message_count = row[3]
                recent_messages = row[4]
                
                # Get the first user message as preview
                preview = ""
                if recent_messages:
                    parts = recent_messages.split(" | ")
                    for part in parts:
                        if part and len(part.strip()) > 0:
                            preview = part.strip()
                            break
                
                threads.append({
                    "thread_id": thread_id,
                    "first_message": first_message,
                    "last_message": last_message,
                    "message_count": message_count,
                    "preview": preview
                })
        
        # If no threads found
        if not threads:
            print("No existing conversation threads found.")
            return None
        
        # Display threads
        print("\nAvailable conversation threads:")
        print("=" * 80)
        
        current_thread_id = load_thread_id(user_id)
        
        for i, thread in enumerate(threads):
            thread_id = thread["thread_id"]
            preview = thread["preview"] if thread["preview"] else "No preview available"
            message_count = thread["message_count"]
            last_message = thread["last_message"].strftime("%Y-%m-%d %H:%M:%S") if thread["last_message"] else "Unknown"
            
            # Mark current thread
            current_marker = " (current)" if thread_id == current_thread_id else ""
            
            print(f"{i+1}. Thread: {thread_id[:8]}...{thread_id[-8:]}{current_marker}")
            print(f"   Last updated: {last_message} ({message_count} messages)")
            print(f"   Preview: {preview[:60]}..." if len(preview) > 60 else f"   Preview: {preview}")
            print("-" * 80)
        
        # Add option for new thread
        print(f"{len(threads)+1}. Create a new conversation thread")
        print("=" * 80)
        
        # Get user selection
        while True:
            try:
                choice = input("Select a thread (enter the number): ")
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(threads):
                    # Return the selected thread ID
                    selected_thread = threads[choice_num-1]["thread_id"]
                    print(f"Selected thread: {selected_thread}")
                    return selected_thread
                elif choice_num == len(threads)+1:
                    # Create a new thread
                    print("Starting a new conversation thread.")
                    return None
                else:
                    print(f"Please enter a number between 1 and {len(threads)+1}")
            except ValueError:
                print("Please enter a valid number")
    
    except Exception as e:
        print(f"Error listing threads: {str(e)}")
        return None

def save_user_data(user_data: dict) -> None:
    """Save the user data to storage."""
    with open(USER_STORAGE, 'w') as f:
        json.dump(user_data, f)


def main():
    """Run the memory agent demo."""
    parser = argparse.ArgumentParser(description="Memory Agent Demo")
    parser.add_argument("--new-thread", action="store_true", help="Start a new conversation thread")
    parser.add_argument("--new-user", action="store_true", help="Start with a new user identity")
    parser.add_argument("--list-threads", action="store_true", help="List and select from available threads")
    args = parser.parse_args()
    
    # Make sure user storage exists
    if not os.path.exists(USER_STORAGE):
        with open(USER_STORAGE, 'w') as f:
            default_user_id = str(uuid.uuid4())
            json.dump({"user_id": default_user_id}, f)
    
    # Load or create user data
    user_data = load_user_data()
    user_id = user_data["user_id"]
    
    if args.new_user:
        user_id = str(uuid.uuid4())
        user_data["user_id"] = user_id
        save_user_data(user_data)
        print(f"Created new user with ID: {user_id}")
    
    # Load existing thread ID or select from list
    thread_id = None
    if args.new_thread:
        thread_id = None
        print("Starting a new conversation thread.")
    elif args.list_threads:
        thread_id = list_and_select_threads(user_id)
    else:
        thread_id = load_thread_id(user_id)
        if thread_id:
            print(f"Continuing previous conversation thread: {thread_id}")
        else:
            print("No existing thread found. Starting a new conversation.")
    
    print("\nMemory Agent Demo")
    print(f"User ID: {user_id}")
    print(f"Thread ID: {thread_id or 'New thread (will be created)'}")
    print("Type 'exit' to quit the conversation.")
    print("Type 'new' to start a new thread.")
    print("Type 'list' to view and select from available threads.")
    print("=" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            # Save the user ID before quitting
            save_user_data({"user_id": user_id})
            print("Goodbye! Your conversation has been saved.")
            break
            
        if user_input.lower() == "new":
            thread_id = None
            print("Starting a new conversation thread.")
            continue
            
        if user_input.lower() == "list":
            thread_id = list_and_select_threads(user_id)
            continue
        
        # Run the agent
        response, new_thread_id = run_agent(user_input, thread_id, user_id)
        
        # Save the thread ID if it changed
        if new_thread_id != thread_id:
            thread_id = new_thread_id
            save_thread_id(user_id, thread_id)
        
        # Display the response
        print(f"\nAgent: {response}")

if __name__ == "__main__":
    main()