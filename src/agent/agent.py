"""
Memory-enabled agent implementation using LangGraph with PostgreSQL.
"""

import os
import json
from typing import Dict, List, Any, Optional, TypedDict, cast
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
from sentence_transformers import SentenceTransformer

from src.model.claude_client import get_claude_client
from src.model.prompts import create_memory_agent_prompt, enhance_prompt_with_memory
from src.memory.memory_manager import ThreadMemory
from src.model.config import get_model_config, get_langgraph_config
from src.agent.schema import (
    AgentState, MemoryData, AgentConfig, MessageType,
    NODE_INITIALIZE, NODE_PROCESS_INPUT, NODE_GENERATE_RESPONSE
)
from src.database.db import get_db_connection
from src.utils.thread_utils import (
    get_all_threads, 
    load_thread_id, 
    save_thread_id,
    MEMORY_DIR
)

# Re-export the functions needed by the API
__all__ = [
    "run_agent", 
    "get_all_threads", 
    "load_thread_id", 
    "save_thread_id", 
    "load_thread_state", 
    "save_thread_state"
]

# Initialize database connection
try:
    conn = get_db_connection()
    print("Connected to PostgreSQL database")
    
    # Initialize our simple saver
    from src.database.simple_saver import SimpleStateSaver
    simple_saver = SimpleStateSaver(conn)
    print("Simple state saver initialized")
    
    # Initialize LangGraph's checkpointer (fallback)
    checkpointer = InMemorySaver()
    
    # Initialize the embedding model for later use
    embedding_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
    
    # Use an in-memory store for user info across threads
    memory_store = InMemoryStore()
    print("Using in-memory store with direct SQL for vectors")
    
    # Verify vectors table exists
    with conn.cursor() as cur:
        # Check if message_vectors table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'message_vectors'
            )
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            # Create the table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS message_vectors (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    role TEXT NOT NULL,
                    embedding vector(768),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Try to create an index
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS message_vectors_embedding_idx 
                    ON message_vectors 
                    USING ivfflat (embedding vector_cosine_ops)
                """)
            except Exception as e:
                print(f"Note: Could not create vector index: {str(e)}")
                print("It's recommended to run fix_schema.py to set up indices properly")
        
        conn.commit()  # Commit the changes

except Exception as e:
    print(f"Error setting up PostgreSQL: {str(e)}")
    print("Falling back to in-memory storage")
    # Fallback to in-memory storage if PostgreSQL setup fails
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    
    checkpointer = InMemorySaver()
    memory_store = InMemoryStore()
    embedding_model = None
    conn = None

# Function to store message vector directly in PostgreSQL
def store_message_vector(user_id, thread_id, content, role):
    """Store a message vector directly in PostgreSQL."""
    if not conn or not embedding_model or not content:
        return
    
    try:
        # Generate embedding
        embedding = embedding_model.encode(content).tolist()
        
        # Store in database with proper vector casting
        with conn.cursor() as cur:
            # Ensure the embedding is properly cast to a vector type
            cur.execute("""
                INSERT INTO message_vectors (user_id, thread_id, content, role, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
            """, (user_id, thread_id, content, role, embedding))
            conn.commit()  # Explicitly commit the transaction
        
        print(f"Stored vector for message in thread {thread_id}")
    except Exception as e:
        print(f"Failed to store message vector: {str(e)}")
        try:
            conn.rollback()  # Rollback on error
        except:
            pass

# Function to search similar messages
def search_similar_messages(user_id, query, limit=3):
    """Search for similar messages using vector similarity."""
    if not embedding_model or not query:
        return []
    
    try:
        # Use a completely separate connection for vector search
        search_conn = get_db_connection()
        
        # Generate embedding for query
        query_embedding = embedding_model.encode(query).tolist()
        
        # Search in database with correct vector similarity operator
        with search_conn.cursor() as cur:
            try:
                # First try with cosine distance (<->)
                cur.execute("""
                    SELECT content, role, thread_id
                    FROM message_vectors
                    WHERE user_id = %s
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                """, (user_id, query_embedding, limit))
                
                results = cur.fetchall()
                search_conn.commit()
            except Exception as e:
                search_conn.rollback()
                print(f"Cosine similarity failed: {str(e)}")
                # Fallback to L2 distance (<=>)
                cur.execute("""
                    SELECT content, role, thread_id
                    FROM message_vectors
                    WHERE user_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (user_id, query_embedding, limit))
                
                results = cur.fetchall()
                search_conn.commit()
            
            memories = []
            for content, role, thread_id in results:
                memories.append({
                    "content": content,
                    "role": role,
                    "thread_id": thread_id
                })
            
            return memories
    except Exception as e:
        print(f"Failed to search similar messages: {str(e)}")
        return []
    finally:
        if 'search_conn' in locals():
            search_conn.close()

# Define namespace format for memory storage
def get_memory_namespace(key: str):
    """Get namespace for memory storage."""
    return (key, "memories")

# Modify run_agent to use PostgreSQL storage
def run_agent(user_input: str, thread_id: Optional[str] = None, user_id: Optional[str] = None):
    """
    Run the memory agent with the given input.
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
    
    # Try to load previous state with our simple saver
    initial_state = {
        "thread_id": current_thread_id,
        "input_text": user_input,
        "messages": [],
        "memory_data": {"user_info": {}, "context": []}
    }
    
    # Try to load previous state
    try:
        if 'simple_saver' in globals() and simple_saver:
            previous_state = simple_saver.load_state(current_thread_id)
            if previous_state:
                # Update with previous state but keep new input
                initial_state["messages"] = previous_state.get("messages", [])
                initial_state["memory_data"] = previous_state.get("memory_data", {"user_info": {}, "context": []})
                print(f"DEBUG: Loaded previous state with {len(initial_state['messages'])} messages and memory data: {initial_state['memory_data']}")
    except Exception as e:
        print(f"DEBUG: Error loading previous state: {str(e)}")
    
    # Also try to load from cross-thread memory if we have user_id
    if user_id and 'memory_store' in globals() and memory_store:
        try:
            namespace = get_memory_namespace(user_id)
            user_info_item = memory_store.get(namespace, "user_info")
            if user_info_item and user_info_item.value:
                initial_state["memory_data"]["user_info"] = user_info_item.value
                print(f"DEBUG: Loaded user info from cross-thread memory: {user_info_item.value}")
        except Exception as e:
            print(f"DEBUG: Error loading from cross-thread memory: {str(e)}")
    
    # Add the new user message to conversation history
    initial_state["messages"].append(HumanMessage(content=user_input))
    
    # Store the message vector
    if user_id and 'store_message_vector' in globals():
        store_message_vector(user_id, current_thread_id, user_input, "user")
    
    # Run the agent
    print(f"DEBUG: Invoking agent with state containing {len(initial_state['messages'])} messages")
    result = agent.invoke(initial_state, config=config)
    
    # Save state with our simple saver
    try:
        if 'simple_saver' in globals() and simple_saver:
            simple_saver.save_state(current_thread_id, result)
            print(f"DEBUG: Saved state with {len(result['messages'])} messages")
    except Exception as e:
        print(f"DEBUG: Error saving state: {str(e)}")
    
    # Store user information in memory store
    if user_id and 'memory_store' in globals() and memory_store and "memory_data" in result and "user_info" in result["memory_data"]:
        user_info = result["memory_data"]["user_info"]
        if user_info:
            # Store in cross-thread memory (user level)
            user_namespace = get_memory_namespace(user_id)
            memory_store.put(user_namespace, "user_info", user_info)
            print(f"DEBUG: Stored user info in cross-thread memory: {user_info}")
    
    # Extract the response (last message content)
    messages = result["messages"]
    if messages and len(messages) > 0:
        last_message = messages[-1]
        response = last_message.content
        
        # Store the response vector
        if user_id and 'store_message_vector' in globals():
            store_message_vector(user_id, current_thread_id, response, "assistant")
    else:
        response = "No response generated."
    
    return response, current_thread_id

# Function to create the agent (this remains mostly the same)
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
    def initialize_state(state: Dict[str, Any], config: Dict[str, Any] = None, *, store=None) -> Dict[str, Any]:
        """Initialize or load the agent state."""
        print(f"DEBUG: initialize_state received keys: {list(state.keys())}")
    
        # Get user_id and thread_id from config if available
        user_id = config["configurable"].get("user_id") if config and "configurable" in config else None
        thread_id = config["configurable"].get("thread_id") if config and "configurable" in config else state.get("thread_id")
    
        # Create a new state with defaults and overrides from input state
        new_state = {
            "thread_id": thread_id or str(uuid.uuid4()),
            "messages": state.get("messages", []),
            "memory_data": state.get("memory_data", {"user_info": {}, "context": []}),
            "input_text": state.get("input_text", "")
        }
    
        # Try to load both user-specific and thread-specific memory
        if user_id and store:
            # 1. Try to load from cross-thread memory (user level)
            try:
                user_namespace = get_memory_namespace(user_id)
                user_info_item = store.get(user_namespace, "user_info")
                if user_info_item and user_info_item.value:
                    print(f"DEBUG: Loaded user info from cross-thread memory: {user_info_item.value}")
                    new_state["memory_data"]["user_info"] = user_info_item.value
            except Exception as e:
                print(f"DEBUG: Error loading user info from cross-thread memory: {str(e)}")
        
            # 2. If we have a thread_id, also try to load from thread-specific memory
            if thread_id:
                try:
                    thread_namespace = get_memory_namespace(f"{user_id}_{thread_id}")
                    thread_info_item = store.get(thread_namespace, "thread_info")
                    if thread_info_item and thread_info_item.value:
                        print(f"DEBUG: Loaded thread-specific info: {thread_info_item.value}")
                        # Override with thread-specific info (more specific than user-level)
                        if "user_info" in thread_info_item.value:
                            new_state["memory_data"]["user_info"] = thread_info_item.value["user_info"]
                except Exception as e:
                    print(f"DEBUG: Error loading thread-specific info: {str(e)}")
        
            # 3. As a backup, try to retrieve from the message vectors
            try:
                if thread_id and user_id:
                    # Get user info from the first few messages in the thread
                    with conn.cursor() as cur:
                        cur.execute("""
                            SELECT content
                            FROM message_vectors
                            WHERE user_id = %s AND thread_id = %s
                            ORDER BY created_at
                            LIMIT 5
                        """, (user_id, thread_id))
                    
                        messages = [row[0] for row in cur.fetchall()]
                    
                        # Create a temporary memory manager to extract info
                        temp_memory = ThreadMemory(thread_id)
                    
                        # Extract from early messages
                        for message in messages:
                            temp_memory.extract_info_from_message(message)
                    
                        # If we found info, use it
                        if temp_memory.user_info:
                            print(f"DEBUG: Recovered user info from message history: {temp_memory.user_info}")
                            new_state["memory_data"]["user_info"] = temp_memory.user_info
            except Exception as e:
                print(f"DEBUG: Error recovering info from message history: {str(e)}")
    
        print(f"DEBUG: initialize_state returned keys: {list(new_state.keys())}")
        return new_state
    
    def process_input(state: Dict[str, Any], config: Dict[str, Any] = None, *, store=None) -> Dict[str, Any]:
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
        
        # We're using direct SQL now, no need for store.put
        
        print(f"DEBUG: process_input returned with user_info: {memory.user_info}")
        return new_state
    
    def generate_response(state: Dict[str, Any], config: Dict[str, Any] = None, *, store=None) -> Dict[str, Any]:
        """Generate a response using Claude with memory context."""
        print(f"DEBUG: generate_response received keys: {list(state.keys())}")
        print(f"DEBUG: Memory data: {state.get('memory_data', {})}")
        print(f"DEBUG: Received {len(state.get('messages', []))} messages")

        new_state = state.copy()

        # Create memory manager to get memory summary
        memory = ThreadMemory(state.get("thread_id", str(uuid.uuid4())))
        memory.user_info = state.get("memory_data", {}).get("user_info", {})
        memory.context = state.get("memory_data", {}).get("context", [])
    
        # If user_info is empty but we have messages, try to extract info
        if not memory.user_info and state.get("messages"):
            # Extract info from previous messages in this thread
            for msg in state.get("messages", []):
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    memory.extract_info_from_message(msg.content)
        
            # Update the state with extracted info
            if memory.user_info:
                print(f"DEBUG: Extracted user info from conversation history: {memory.user_info}")
                new_state["memory_data"]["user_info"] = memory.user_info
    
        # Get memory context from thread-specific memory
        memory_context = memory.get_memory_summary()
        
        # Get additional context from vector store if available
        additional_context = ""
        if config and "configurable" in config:
            user_id = config["configurable"].get("user_id")
            if user_id and state.get("input_text"):
                # Use our direct SQL search
                memories = search_similar_messages(user_id, state.get("input_text"), limit=3)
                
                if memories:
                    additional_context = "\nRelevant past information:\n"
                    for memory_item in memories:
                        if memory_item.get("content") and memory_item.get("role") == "user":
                            additional_context += f"- User said: {memory_item['content']}\n"
        
        # Combine thread memory with vector search results
        if additional_context:
            memory_context = f"{memory_context}\n{additional_context}"
        
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
        
        # We're using direct SQL now, no need for store.put
        
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

# Legacy functions for backward compatibility (can be removed later)
def load_thread_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """Load thread state using the PostgreSQL checkpointer."""
    # This now just serves as a compatibility layer
    # Real state loading is handled by LangGraph's PostgresSaver
    
    # Create a temporary graph to access the state
    agent = create_memory_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Try to get state using LangGraph's built-in methods
        state_snapshot = agent.get_state(config)
        if state_snapshot:
            return state_snapshot.values
    except Exception as e:
        print(f"DEBUG: Error loading state from PostgreSQL: {str(e)}")
    
    # Fallback to file-based method if needed
    file_path = os.path.join(MEMORY_DIR, f"{thread_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"DEBUG: Error loading thread state from file: {str(e)}")
    
    return None

def save_thread_state(thread_id: str, state: Dict[str, Any]) -> None:
    """Save thread state (compatibility method, not needed with PostgreSQL)."""
    # This is now just a no-op as PostgreSQL checkpointer handles saving automatically
    # Keep for backward compatibility with code that might call this
    pass
