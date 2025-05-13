"""
Schema definitions for the memory agent.
"""

from typing import Dict, List, Any, Optional, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Define message types for better type hinting
MessageType = Union[HumanMessage, AIMessage, SystemMessage]

class AgentState(TypedDict):
    """
    State schema for the memory agent.
    
    This defines the structure of the state that is maintained
    throughout the agent's execution and persisted in threads.
    """
    # Conversation history
    messages: List[MessageType]
    
    # Thread identifier for persistence
    thread_id: str
    
    # Memory management data
    memory_data: Dict[str, Any]
    
    # Input text from user (optional as it may not always be present)
    input_text: Optional[str]

class MemoryData(TypedDict):
    """
    Structure for memory data stored within the agent state.
    """
    # User information (name, preferences, etc.)
    user_info: Dict[str, str]
    
    # Additional context information
    context: List[str]

class ConfigurableOptions(TypedDict):
    """
    Configurable options that can be passed to the agent.
    """
    # Thread ID for persistence
    thread_id: str
    
    # Optional user ID for cross-thread memory
    user_id: Optional[str]
    
    # Optional checkpoint ID for time travel
    checkpoint_id: Optional[str]

class AgentConfig(TypedDict):
    """
    Configuration for the memory agent.
    """
    configurable: ConfigurableOptions

# Define constants for node names
NODE_INITIALIZE = "initialize"
NODE_PROCESS_INPUT = "process_input"
NODE_GENERATE_RESPONSE = "generate_response"