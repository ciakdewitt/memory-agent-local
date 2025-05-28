"""
Command-line interface for the memory agent.
"""

import os
import json
from typing import Optional, Tuple
import datetime

from src.agent.agent import run_agent

class MemoryAgentCLI:
    """Simple CLI for interacting with the memory agent."""
    
    def __init__(self, thread_storage_path: str = "thread_storage.json"):
        """Initialize the CLI."""
        self.thread_storage_path = thread_storage_path
        
    def load_thread_id(self) -> Optional[str]:
        """Load the thread ID from storage if it exists."""
        if os.path.exists(self.thread_storage_path):
            with open(self.thread_storage_path, 'r') as f:
                data = json.load(f)
                return data.get("thread_id")
        return None
    
    def save_thread_id(self, thread_id: str) -> None:
        """Save the thread ID to storage."""
        with open(self.thread_storage_path, 'w') as f:
            json.dump({
                "thread_id": thread_id,
                "last_used": datetime.datetime.now().isoformat()
            }, f)
    
    def print_welcome(self, thread_id: Optional[str] = None) -> None:
        """Print welcome message."""
        print("\n" + "=" * 50)
        print("Memory Agent Demo")
        print("=" * 50)
        
        if thread_id:
            print("Continuing previous conversation thread.")
        else:
            print("Starting a new conversation thread.")
            
        print("\nType 'exit' to quit or 'new' to start a new thread.")
        print("=" * 50)
    
    def run_interactive_session(self, new_thread: bool = False) -> None:
        """Run an interactive session with the memory agent."""
        # Load existing thread ID unless new_thread flag is True
        thread_id = None if new_thread else self.load_thread_id()
        
        self.print_welcome(thread_id)
        
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nGoodbye!")
                break
                
            if user_input.lower() == "new":
                print("\nStarting a new conversation thread.")
                thread_id = None
                continue
            
            # Run the agent
            response, thread_id = run_agent(user_input, thread_id)
            
            # Save the thread ID
            self.save_thread_id(thread_id)
            
            # Display the response
            print(f"\nAgent: {response}")