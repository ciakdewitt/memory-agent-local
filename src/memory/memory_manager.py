"""
Memory management implementation for thread-based conversation memory.
"""

from typing import Dict, List, Any, Optional

class ThreadMemory:
    """
    A class to manage memory within a thread context.
    
    This implements short-term memory for the agent by storing
    user information and context within a thread.
    """
    
    def __init__(self, thread_id: str):
        """
        Initialize the memory for a specific thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
        """
        self.thread_id = thread_id
        self.user_info = {}  # Store user details like name, preferences, etc.
        self.context = []    # Store additional context
        
    def update_user_info(self, key: str, value: Any) -> None:
        """
        Update user information in memory.
        
        Args:
            key: The attribute name (e.g., "name", "preference")
            value: The value to store
        """
        self.user_info[key] = value
        
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get all stored user information.
        
        Returns:
            Dict containing user information
        """
        return self.user_info
    
    def add_context(self, context_item: Any) -> None:
        """
        Add a context item to memory.
        
        Args:
            context_item: Information to add to context
        """
        self.context.append(context_item)
    
    def get_memory_summary(self) -> str:
        """
        Generate a summary of the memory for inclusion in prompts.
        
        Returns:
            String summary of relevant memory
        """
        summary = []
        
        if self.user_info:
            user_info_str = ", ".join([f"{k}: {v}" for k, v in self.user_info.items()])
            summary.append(f"User information: {user_info_str}")
            
        if self.context:
            context_str = "; ".join(str(item) for item in self.context)
            summary.append(f"Additional context: {context_str}")
            
        return "\n".join(summary)
    
    def extract_info_from_message(self, message: str) -> None:
        """
        Extract information from a user message to update memory.
        
        This is a simple implementation that could be expanded with
        more sophisticated NLP techniques.
        
        Args:
            message: The user's message text
        """
        # Extract name if mentioned
        message_lower = message.lower()
        
        # Simple name extraction
        if "my name is" in message_lower:
            name_part = message_lower.split("my name is")[1].strip()
            # Get the first word after "my name is"
            name = name_part.split()[0].capitalize()
            self.update_user_info("name", name)
            
        # Extract preferences if mentioned
        if "i like" in message_lower:
            like_part = message_lower.split("i like")[1].strip()
            # Get the phrase after "I like"
            preference = like_part.split(".")[0].strip()
            self.update_user_info("likes", preference)
            
        # Extract dislikes if mentioned
        if "i don't like" in message_lower or "i dont like" in message_lower:
            if "i don't like" in message_lower:
                dislike_part = message_lower.split("i don't like")[1].strip()
            else:
                dislike_part = message_lower.split("i dont like")[1].strip()
            # Get the phrase after "I don't like"
            dislike = dislike_part.split(".")[0].strip()
            self.update_user_info("dislikes", dislike)