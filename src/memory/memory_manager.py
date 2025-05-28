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
    
        Enhanced version that recognizes more patterns.
    
        Args:
            message: The user's message text
        """
        message_lower = message.lower()
    
        # Extract name if mentioned
        if "my name is" in message_lower:
            name_part = message_lower.split("my name is")[1].strip()
            # Get the first word after "my name is" - remove punctuation
            name = name_part.split()[0].rstrip(',.:;!?')
            self.update_user_info("name", name.capitalize())
        elif "i am " in message_lower and len(message_lower.split("i am ")) > 1:
            # Try to extract name from "I am [Name]" pattern
            name_part = message_lower.split("i am ")[1].strip()
            if not any(job in name_part for job in ["developer", "engineer", "student", "working", "employee"]):
                name = name_part.split()[0].rstrip(',.:;!?')
                if name and len(name) > 2:  # Avoid short words that aren't names
                    self.update_user_info("name", name.capitalize())
    
        # Extract job/occupation - improved pattern matching
        job_patterns = [
            "i am a", "i'm a", 
            "i am an", "i'm an",
            "i work as", "my job is", 
            "my profession is"
        ]
    
        for pattern in job_patterns:
            if pattern in message_lower:
                job_part = message_lower.split(pattern)[1].strip()
                if "." in job_part:
                    job = job_part.split(".")[0].strip()
                elif "," in job_part:
                    job = job_part.split(",")[0].strip()
                else:
                    job = job_part.split()[0] + " " + (job_part.split()[1] if len(job_part.split()) > 1 else "")
            
                self.update_user_info("occupation", job)
                break
    
        # Special case for direct occupation statements
        if "i am software engineer" in message_lower or "i am an ai engineer" in message_lower:
            if "ai engineer" in message_lower:
                self.update_user_info("occupation", "AI Engineer")
            else:
                self.update_user_info("occupation", "Software Engineer")