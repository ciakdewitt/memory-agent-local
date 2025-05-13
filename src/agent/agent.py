"""
Memory-enabled agent implementation using LangGraph.
"""

import os
import json
import sqlite3
import threading
from typing import Dict, List, Any, Optional, TypedDict, cast
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.model.claude_client import get_claude_client
from src.model.prompts import create_memory_agent_prompt, enhance_prompt_with_memory
from src.memory.memory_manager import ThreadMemory
from src.model.config import get_model_config, get_langgraph_config
from src.agent.schema import (
    AgentState, MemoryData, AgentConfig, MessageType,
    NODE_INITIALIZE, NODE_PROCESS_INPUT, NODE_GENERATE_RESPONSE
)

# Initialize the checkpointer
print("Using InMemorySaver with custom file persistence")
checkpointer = InMemorySaver()

# Initialize the memory store
memory_store = InMemoryStore()

# Define memory persistence directory
MEMORY_DIR = "./.memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

def save_thread_state(thread_id: str, state: Dict[str, Any]) -> None:
    """Save thread state to a file for persistence between sessions."""
    file_path = os.path.join(MEMORY_DIR, f"{thread_id}.json")
    
    # Extract the parts of state we want to save
    serializable_state = {
        "thread_id": state["thread_id"],
        "memory_data": state["memory_data"],
        "messages": []
    }
    
    # Convert message objects to serializable form
    for msg in state.get("messages", []):
        msg_type = type(msg).__name__
        content = msg.content
        
        serializable_state["messages"].append({
            "type": msg_type,
            "content": content
        })
    
    # Save to file
    with open(file_path, 'w') as f:
        json.dump(serializable_state, f)
    
    print(f"DEBUG: Saved thread state to {file_path}")

def load_thread_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """Load thread state from a file."""
    file_path = os.path.join(MEMORY_DIR, f"{thread_id}.json")
    
    if not os.path.exists(file_path):
        print(f"DEBUG: No saved state found for thread {thread_id}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            serializable_state = json.load(f)
        
        # Convert serialized messages back to message objects
        messages = []
        for msg_data in serializable_state.get("messages", []):
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            
            if msg_type == "HumanMessage":
                messages.append(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                messages.append(AIMessage(content=content))
            elif msg_type == "SystemMessage":
                messages.append(SystemMessage(content=content))
        
        # Create state dictionary
        state = {
            "thread_id": serializable_state.get("thread_id"),
            "memory_data": serializable_state.get("memory_data", {"user_info": {}, "context": []}),
            "messages": messages
        }
        
        print(f"DEBUG: Loaded thread state from {file_path}")
        print(f"DEBUG: Loaded {len(messages)} messages and memory data with keys: {list(state['memory_data'].keys())}")
        
        return state
    except Exception as e:
        print(f"DEBUG: Error loading thread state: {str(e)}")
        return None

def create_memory_agent():
    """
    Create a memory-enabled agent using LangGraph.
    
    Returns:
        A runnable agent graph
    """
    # Initialize the language model
    claude = get_claude_client()
    base_system_prompt = create_memory_agent_prompt()
    
    # Define the agent nodes
    def initialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize or load the agent state."""
        print(f"DEBUG: initialize_state received keys: {list(state.keys())}")
    
        # Create a new state with defaults and overrides from input state
        new_state = {
            "thread_id": state.get("thread_id", str(uuid.uuid4())),
            "messages": state.get("messages", []),
            "memory_data": state.get("memory_data", {"user_info": {}, "context": []}),
            "input_text": state.get("input_text", "")
        }
    
        print(f"DEBUG: initialize_state returned keys: {list(new_state.keys())}")
        return new_state
    
    def process_input(state: Dict[str, Any]) -> Dict[str, Any]:
        """Process user input and update memory if needed."""
        print(f"DEBUG: process_input received keys: {list(state.keys())}")
    
        # Get the input text
        input_text = state.get("input_text", "")
        if not input_text:
            print("DEBUG: No input text found in state")
            return state  # No input to process
        
        # Create new state to avoid modifying the original
        new_state = state.copy()
    
        # Add the user message to conversation history
        new_state["messages"] = state.get("messages", []) + [HumanMessage(content=input_text)]
    
        # Create memory manager to extract information
        memory = ThreadMemory(state.get("thread_id", str(uuid.uuid4())))
        memory.user_info = state.get("memory_data", {}).get("user_info", {})
        memory.context = state.get("memory_data", {}).get("context", [])
    
        # Extract information from the message
        memory.extract_info_from_message(input_text)
    
        # Update memory in the state
        new_state["memory_data"] = {
            "user_info": memory.user_info,
            "context": memory.context
        }
    
        print(f"DEBUG: process_input returned with user_info: {memory.user_info}")
        return new_state
    
    def generate_response(state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response using Claude with memory context."""
        print(f"DEBUG: generate_response received keys: {list(state.keys())}")
    
        new_state = state.copy()
    
        # Create memory manager to get memory summary
        memory = ThreadMemory(state.get("thread_id", str(uuid.uuid4())))
        memory.user_info = state.get("memory_data", {}).get("user_info", {})
        memory.context = state.get("memory_data", {}).get("context", [])
    
        # Get memory context
        memory_context = memory.get_memory_summary()
        print(f"DEBUG: Memory context: {memory_context}")
    
        # Use the enhance_prompt_with_memory function 
        enhanced_system_prompt = enhance_prompt_with_memory(base_system_prompt, memory_context)
        
        model_messages = [SystemMessage(content=enhanced_system_prompt)]
    
        # Add conversation history
        for msg in state.get("messages", []):
            model_messages.append(msg)
    
        print(f"DEBUG: Sending {len(model_messages)} messages to Claude")
    
        # Get response from Claude
        response = claude.invoke(model_messages)
    
        # Add the assistant's response to the conversation
        new_state["messages"] = state.get("messages", []) + [AIMessage(content=response.content)]
    
        return new_state
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node(NODE_INITIALIZE, initialize_state)
    workflow.add_node(NODE_PROCESS_INPUT, process_input)
    workflow.add_node(NODE_GENERATE_RESPONSE, generate_response)
    
    # Add edges
    workflow.set_entry_point(NODE_INITIALIZE)
    workflow.add_edge(NODE_INITIALIZE, NODE_PROCESS_INPUT)
    workflow.add_edge(NODE_PROCESS_INPUT, NODE_GENERATE_RESPONSE)
    workflow.add_edge(NODE_GENERATE_RESPONSE, END)
    
    # Compile the graph with both checkpointer and store
    memory_agent = workflow.compile(checkpointer=checkpointer, store=memory_store)
    
    return memory_agent

def run_agent(user_input: str, thread_id: Optional[str] = None, user_id: Optional[str] = None):
    """
    Run the memory agent with the given input.
    
    Args:
        user_input: The user's message
        thread_id: Optional thread ID for continuing a conversation
        user_id: Optional user ID for cross-thread memory
        
    Returns:
        The agent's response and the thread ID
    """
    agent = create_memory_agent()
    
    # Use the provided thread_id or generate a new one
    current_thread_id = thread_id or str(uuid.uuid4())
    print(f"DEBUG: Running agent with thread_id={current_thread_id}")
    
    # Prepare config with thread_id in the configurable section
    config = {"configurable": {"thread_id": current_thread_id}}
    
    # Add user_id to config if provided
    if user_id:
        config["configurable"]["user_id"] = user_id
    
    # Create initial state
    initial_state = {
        "thread_id": current_thread_id,
        "input_text": user_input,
        "messages": [],
        "memory_data": {"user_info": {}, "context": []}
    }
    
    # Try to load existing state from our file storage
    if thread_id:
        loaded_state = load_thread_state(thread_id)
        if loaded_state:
            # Merge the loaded state with our initial state
            initial_state["messages"] = loaded_state.get("messages", [])
            initial_state["memory_data"] = loaded_state.get("memory_data", initial_state["memory_data"])
            print(f"DEBUG: Using loaded state for thread {thread_id} with {len(initial_state['messages'])} messages")
    
    # Run the agent
    print(f"DEBUG: Invoking agent with input: {initial_state}")
    result = agent.invoke(initial_state, config=config)
    print(f"DEBUG: Agent result keys: {list(result.keys())}")
    
    # Save the updated state to our file storage
    save_thread_state(current_thread_id, result)
    
    # Extract the response (last message content)
    messages = result["messages"]
    if messages and len(messages) > 0:
        last_message = messages[-1]
        response = last_message.content
    else:
        response = "No response generated."
    
    return response, current_thread_id