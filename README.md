# Memory Agent Demo

A demonstration of short-term memory capabilities using LangGraph thread-based persistence.

## Features

- Conversational agent that remembers information across sessions
- Extracts and stores personal information shared during conversations
- Thread-based persistence for maintaining conversation context
- Uses Claude's LLM capabilities for natural interactions
- Custom file-based persistence layer for reliable memory storage
- Simple command-line interface for easy interaction

## Architecture

The Memory Agent is built using several key components:

1. **LangGraph Framework**: For defining the agent's workflow and state management
2. **Claude LLM**: For natural language understanding and generation
3. **Custom Persistence Layer**: For storing conversation state between sessions
4. **Memory Manager**: For extracting and managing user information

The agent maintains information in two main levels:
- **Short-term Memory**: The full conversation history for a given thread
- **Extracted Information**: Specific details extracted from conversations (like names, preferences, etc.)

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv myxagent`
3. Activate the environment:
   - Windows: `myxagent\Scripts\activate`
   - macOS/Linux: `source myxagent/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
5. Create a `.env` file with your Anthropic API key:


## Project Structure

- `src/agent/`: Agent implementation using LangGraph
- `agent.py`: Main agent implementation with memory persistence
- `schema.py`: Type definitions and state schema
- `src/memory/`: Memory management components
- `memory_manager.py`: Memory extraction and management
- `src/model/`: Model integration and configuration
- `claude_client.py`: Claude LLM integration
- `config.py`: Configuration management
- `prompts.py`: System prompts for the agent
- `.memory/`: Directory for storing persisted memory files
- `main.py`: Command-line interface for the memory agent

## Memory Persistence

The agent uses a hybrid approach to memory persistence:

1. **InMemorySaver**: For within-session state management with LangGraph
2. **Custom File-based Persistence**: For cross-session persistence

Each conversation thread is stored in a JSON file in the `.memory` directory, containing:
- Full conversation history (messages)
- Extracted user information
- Contextual data

This enables the agent to remember information across sessions, creating a more natural and personalized interaction experience.

## Usage

### Running the Agent

```bash
python main.py
```

### This will:

Start the memory agent in interactive mode
Load an existing conversation thread if one exists
Create a new thread automatically if no previous thread is found
Store information shared during the conversation (like your name, preferences, etc.)
Save the thread ID for future sessions

Starting a New Conversation
bashpython main.py --new-thread


Starting a New Conversation
bashpython main.py --new-thread
This will:

Force the creation of a new conversation thread
Ignore any existing saved thread
Start with no memory of previous interactions
Use this when you want to begin a completely fresh conversation

Creating a New User Identity
bashpython main.py --new-user
This will:

Create a new user ID
Start a fresh conversation with no previous history
Use this when you want to simulate a completely different user

Thread Persistence Explained

Thread ID: Each conversation has a unique thread ID that links all interactions
Persistence: The thread ID is saved to thread_storage.json after each session
Memory Storage: Conversation state is saved to .memory/{thread_id}.json
Memory Continuity:

Without --new-thread, the agent will remember details from previous sessions
With --new-thread, the agent starts fresh with no memory of previous interactions



Example Session
In your first session:
You: Hi, my name is Alex
Agent: Nice to meet you Alex! How can I help you today?
In a later session (same thread):
You: Do you remember my name?
Agent: Of course I do, Alex! How can I help you today?
During Conversation

Type your messages and press Enter to send
Type exit, quit, or bye to end the conversation
Type new during a conversation to start a fresh thread immediately

Memory Extraction
The agent automatically extracts information from conversations including:

Names (e.g., "My name is Alex")
Preferences (e.g., "I like pizza")
Dislikes (e.g., "I don't like coffee")

This extracted information is then incorporated into future conversations, allowing the agent to refer to the user by name and remember key details about them.


Implementation Details
Custom Persistence Layer
Instead of relying solely on LangGraph's built-in persistence, we implemented a custom file-based system to ensure reliable cross-session memory:
pythondef save_thread_state(thread_id: str, state: Dict[str, Any]) -> None:
    """Save thread state to a file for persistence between sessions."""
    file_path = os.path.join(MEMORY_DIR, f"{thread_id}.json")
    # ... serialization logic ...
    
def load_thread_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """Load thread state from a file."""
    file_path = os.path.join(MEMORY_DIR, f"{thread_id}.json")
    # ... deserialization logic ...
This approach avoids threading issues and provides more control over the stored data.
Memory Manager
The ThreadMemory class handles information extraction and organization:
pythonclass ThreadMemory:
    """A class to manage memory within a thread context."""
    
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self.user_info = {}  # Store user details
        self.context = []    # Store context
        
    def extract_info_from_message(self, message: str) -> None:
        """Extract information from user messages."""
        # ... extraction logic ...
Agent State
The agent maintains a structured state that captures all relevant information:
pythonclass AgentState(TypedDict):
    """State schema for the memory agent."""
    messages: List[MessageType]  # conversation history
    thread_id: str               # thread identifier
    memory_data: Dict[str, Any]  # memory management data
    input_text: Optional[str]    # input text from user
Limitations and Future Work

Currently only extracts basic information (names, likes, dislikes)
No semantic search capabilities for memory retrieval
Limited cross-thread memory (information sharing between different conversations)
No long-term memory summarization for very long conversations

Future enhancements could include:

More sophisticated information extraction
Vector-based memory retrieval for more relevant context
Memory consolidation for long conversations
Integration with external knowledge bases