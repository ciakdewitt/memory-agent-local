"""
Main application entry point for the Memory Agent demo.
"""

import os
import argparse
from typing import Optional
import json
import uuid

from src.agent.agent import run_agent

# File to store user information
USER_STORAGE = "user_storage.json"
THREAD_STORAGE = "thread_storage.json"

def load_user_data() -> dict:
    """Load the user data from storage if it exists."""
    if os.path.exists(USER_STORAGE):
        with open(USER_STORAGE, 'r') as f:
            return json.load(f)
    return {"user_id": str(uuid.uuid4())}

def load_thread_id(user_id: str) -> Optional[str]:
    """Load the thread ID from storage if it exists."""
    if os.path.exists(THREAD_STORAGE):
        with open(THREAD_STORAGE, 'r') as f:
            try:
                data = json.load(f)
                thread_id = data.get(user_id)
                print(f"DEBUG: Loaded thread ID {thread_id} for user {user_id}")
                return thread_id
            except json.JSONDecodeError:
                print(f"DEBUG: Error loading thread storage - invalid JSON")
    else:
        print(f"DEBUG: Thread storage file does not exist")
    return None

def save_user_data(user_data: dict) -> None:
    """Save the user data to storage."""
    with open(USER_STORAGE, 'w') as f:
        json.dump(user_data, f)
    print(f"DEBUG: Saved user data: {user_data}")

def save_thread_id(user_id: str, thread_id: str) -> None:
    """Save the thread ID to storage."""
    data = {}
    if os.path.exists(THREAD_STORAGE):
        with open(THREAD_STORAGE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"DEBUG: Error reading thread storage - invalid JSON")
                data = {}
    
    data[user_id] = thread_id
    
    with open(THREAD_STORAGE, 'w') as f:
        json.dump(data, f)
    print(f"DEBUG: Saved thread ID {thread_id} for user {user_id}")

# In the main function
def main():
    """Run the memory agent demo."""
    parser = argparse.ArgumentParser(description="Memory Agent Demo")
    parser.add_argument("--new-thread", action="store_true", help="Start a new conversation thread")
    parser.add_argument("--new-user", action="store_true", help="Start with a new user identity")
    args = parser.parse_args()
    
    # Make sure user storage exists
    if not os.path.exists(USER_STORAGE):
        with open(USER_STORAGE, 'w') as f:
            default_user_id = str(uuid.uuid4())
            json.dump({"user_id": default_user_id}, f)
            print(f"Created new user storage with ID: {default_user_id}")
    
    # Load or create user data
    user_data = load_user_data()
    user_id = user_data["user_id"]
    
    if args.new_user:
        user_id = str(uuid.uuid4())
        user_data["user_id"] = user_id
        save_user_data(user_data)
        print(f"Created new user with ID: {user_id}")
    
    # Load existing thread ID unless --new-thread flag is used
    thread_id = None if args.new_thread else load_thread_id(user_id)
    
    if args.new_thread:
        print("Starting a new conversation thread.")
    elif thread_id:
        print(f"Continuing previous conversation thread: {thread_id}")
    else:
        print("No existing thread found. Starting a new conversation.")
    
    print("\nMemory Agent Demo")
    print(f"User ID: {user_id}")
    print("Type 'exit' to quit the conversation.")
    print("Type 'new' to start a new thread.")
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
        
        # Run the agent
        print(f"DEBUG: Running agent with thread_id={thread_id}, user_id={user_id}")
        response, new_thread_id = run_agent(user_input, thread_id, user_id)
        
        # Save the thread ID if it changed
        if new_thread_id != thread_id:
            thread_id = new_thread_id
            save_thread_id(user_id, thread_id)
            print(f"DEBUG: Updated thread ID to {thread_id}")
        
        # Display the response
        print(f"\nAgent: {response}")

if __name__ == "__main__":
    main()