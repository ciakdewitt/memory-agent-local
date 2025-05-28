# Memory Agent Demo

A demonstration of short-term and long-term memory capabilities using LangGraph, PostgreSQL with pgvector, and vector embeddings.

## Features

- Conversational agent that remembers information across sessions
- Semantic memory search using pgvector for better recall of relevant information
- PostgreSQL-based persistence for reliability and scalability
- Cross-thread memory sharing for consistent user experience
- Vector embeddings for understanding the meaning behind conversations
- Thread management interface for organizing multiple conversations
- Simple command-line interface for easy interaction

## Architecture

The Memory Agent is built using several key components:

1. **LangGraph Framework**: For defining the agent's workflow and state management
2. **Claude LLM**: For natural language understanding and generation
3. **PostgreSQL with pgvector**: For persistent storage with vector search capabilities
4. **Sentence Transformers**: For generating text embeddings for semantic search
5. **Memory Manager**: For extracting and managing user information

The agent maintains information in multiple levels:
- **Short-term Memory**: The conversation history within a given thread
- **Cross-thread Memory**: User information that persists across different conversation threads
- **Vector Memory**: Semantic embeddings of past messages for contextual retrieval

## Technical Implementation

### Memory Architecture

The system implements a multi-layered memory architecture:

1. **Thread-Specific Memory**: Stores the complete conversation history and context for a specific thread
   - Implemented using `SimpleStateSaver` for PostgreSQL persistence
   - Each thread maintains its own independent conversation state

2. **Cross-Thread Memory**: Maintains user information across different conversations
   - Stores extracted information like name, preferences, occupation
   - Available even when starting new conversation threads
   - Implemented with `InMemoryStore` and namespace partitioning

3. **Vector Memory**: Semantic search capabilities for retrieving contextually relevant past messages
   - Uses pgvector for storing and searching embeddings
   - Implemented with the SentenceTransformer model
   - Allows the agent to find relevant past conversation regardless of exact wording

### PostgreSQL with pgvector

The system uses PostgreSQL with the pgvector extension for storing and retrieving conversation data:

- **message_vectors table**: Stores message text and vector embeddings for semantic search
- **simple_thread_states table**: Maintains conversation state across sessions
- **user_threads table**: Tracks which thread is active for each user

Vector embeddings enable the agent to retrieve semantically similar past messages, even when exact keywords don't match. This creates a more natural memory retrieval system.

### Memory Management

The `ThreadMemory` class extracts information from conversations:
- Names (e.g., "My name is Edmond")
- Occupations (e.g., "I am an AI Engineer")
- Preferences (e.g., "I like sailing")
- Contextual information (e.g., location, background)

This information is stored in the database and incorporated into future conversations, available across different threads.

### Message Processing Flow

1. **Input Reception**: The user's message is received via the CLI interface
2. **State Initialization**: Previous conversation state is loaded or a new one is created
3. **Information Extraction**: The `ThreadMemory` extracts structured information from the message
4. **Vector Embedding**: The message is embedded and stored in the vector database
5. **Context Retrieval**: Relevant past messages are retrieved using vector similarity search
6. **Memory-Enhanced Response**: The agent generates a response using the conversation history, extracted information, and relevant past messages
7. **State Persistence**: The updated state is saved for future sessions

### Database Schema

#### message_vectors
- `id`: Serial primary key
- `user_id`: User identifier
- `thread_id`: Conversation thread identifier
- `content`: Message text content
- `role`: Message role (user/assistant)
- `embedding`: Vector representation (768 dimensions)
- `created_at`: Timestamp

#### simple_thread_states
- `thread_id`: Thread identifier (primary key)
- `state`: JSON blob containing thread state
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

#### user_threads
- `user_id`: User identifier (primary key)
- `value`: Active thread ID
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for PostgreSQL)
- Anthropic API key (for Claude)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/memoryagent.git
cd memoryagent
```

### 2. Start PostgreSQL with pgvector

```bash
docker-compose up -d db
```

This starts:
- PostgreSQL with pgvector extension at port 5432

### 3. Create a virtual environment

```bash
python -m venv mymemory
source mymemory/bin/activate  # On Windows: mymemory\Scripts\activate
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

### 5. Set up environment variables

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_api_key_here
MODEL_NAME=claude-3-opus
TEMPERATURE=0.7
MAX_TOKENS=4096
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/memory_agent
```

### 6. Initialize the database

```bash
python fix_schema.py
```

This will:
- Create the necessary database tables
- Set up the pgvector extension
- Create indices for vector similarity search

## Usage

### Running the CLI Agent

```bash
python main.py
```

This will:
- Connect to PostgreSQL for persistence
- Load an existing conversation thread if one exists
- Create a new thread automatically if no previous thread is found
- Store information with vector embeddings for semantic retrieval

### Starting a New Conversation

```bash
python main.py --new-thread
```

This will:
- Force the creation of a new conversation thread
- Ignore any existing saved thread
- Start with no thread-specific memory of previous interactions
- Cross-thread memory (like user information) will still be available

### Managing Threads

```bash
python main.py --list-threads
```

This will:
- Show a list of all your conversation threads
- Display a preview of the last message in each thread
- Show the number of messages and last update time
- Allow you to select which thread to continue

### In-Conversation Commands

During a conversation, you can use these commands:
- `list`: View and select from available threads
- `new`: Start a fresh conversation thread
- `exit`: End the session and save the conversation

## Database Querying and Monitoring

You can use pgAdmin or direct SQL queries to examine the conversations and memory states stored in the database. Here are some useful queries to analyze and debug your memory agent:

### View All Messages in a Thread

To see all messages in a specific conversation thread:

```sql
SELECT id, content, role, created_at
FROM message_vectors
WHERE thread_id = 'd6e54d45-3980-426f-9572-4b9bc399fef9'
ORDER BY created_at;
```

Replace the thread_id with the one you want to examine. This shows the complete conversation flow in chronological order.

### Find Similar Messages Using Vector Search

This query performs a similarity search to find messages related to a specific topic:

```sql
-- First, get an embedding for a message about the topic of interest
WITH sample AS (
    SELECT embedding 
    FROM message_vectors 
    WHERE content LIKE '%sailing%' 
    LIMIT 1
)
-- Then find similar messages
SELECT substring(content, 1, 100) as preview, role, thread_id,
       embedding <-> (SELECT embedding FROM sample) as distance
FROM message_vectors
ORDER BY distance
LIMIT 5;
```

This example finds messages similar to ones containing "sailing" using vector similarity.

### View Thread States

To examine the saved state of a specific thread:

```sql
SELECT thread_id, 
       state->'memory_data'->'user_info' as user_info,
       jsonb_array_length(state->'messages') as message_count,
       created_at, updated_at
FROM simple_thread_states
WHERE thread_id = 'd6e54d45-3980-426f-9572-4b9bc399fef9';
```

This shows the extracted user information and conversation statistics for a specific thread.

### Find All Threads for a User

To see all threads belonging to a specific user:

```sql
SELECT DISTINCT thread_id, 
       MIN(created_at) as first_message, 
       MAX(created_at) as last_message,
       COUNT(*) as message_count
FROM message_vectors
WHERE user_id = 'bd54eac5-1022-4a3b-984d-26fc1f1585ce'
GROUP BY thread_id
ORDER BY last_message DESC;
```

### Extract User Information Across Threads

To see what information has been extracted about users:

```sql
SELECT thread_id, 
       state->'memory_data'->'user_info'->>'name' as name,
       state->'memory_data'->'user_info'->>'occupation' as occupation
FROM simple_thread_states
ORDER BY updated_at DESC;
```

This shows the names and occupations extracted from different conversations.

### Find Messages Mentioning Specific Topics

To search for messages containing specific keywords:

```sql
SELECT thread_id, substring(content, 1, 100) as preview, 
       role, created_at
FROM message_vectors
WHERE content ILIKE '%sailing%' OR content ILIKE '%boat%'
ORDER BY created_at DESC;
```

### Monitor Vector Index Performance

To check if your vector index is being used efficiently:

```sql
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'message_vectors';

EXPLAIN ANALYZE
SELECT id, content
FROM message_vectors
ORDER BY embedding <-> '[0.1,0.2,...]'::vector
LIMIT 5;
```

Replace the vector with an actual embedding for accurate analysis.

## Implementation Details

### Memory Extraction

The system uses pattern matching and contextual analysis to extract information from user messages. For example:

```python
# Extract name from "My name is X" pattern
if "my name is" in message_lower:
    name_part = message_lower.split("my name is")[1].strip()
    name = name_part.split()[0].rstrip(',.:;!?')
    self.update_user_info("name", name.capitalize())
```

### Vector Similarity Search

Vector similarity search uses cosine similarity to find related messages:

```python
# Search for similar messages using vector similarity
cur.execute("""
    SELECT content, role, thread_id
    FROM message_vectors
    WHERE user_id = %s
    ORDER BY embedding <-> %s::vector
    LIMIT %s
""", (user_id, query_embedding, limit))
```

### State Management

Thread state is saved using a custom state saver that handles LangGraph message objects:

```python
def save_state(self, thread_id: str, state: Dict[str, Any]) -> None:
    # Convert state to serializable format
    serializable_state = self._prepare_state_for_storage(state)
    
    # Store in database
    cur.execute("""
        INSERT INTO simple_thread_states (thread_id, state, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (thread_id) 
        DO UPDATE SET state = %s, updated_at = NOW()
    """, (thread_id, json.dumps(serializable_state), json.dumps(serializable_state)))
```

## Implementation Challenges and Solutions

During development, we encountered and solved several challenges:

1. **PostgreSQL Pipeline Mode Issues**: Errors with PostgreSQL's pipeline mode in LangGraph's PostgresStore. Solved by creating a custom saver that disables pipeline mode.

2. **Vector Type Casting**: Fixed proper vector type casting for pgvector operations using `::vector` syntax.

3. **Cross-Thread Memory**: Implemented robust user information extraction and retrieval to ensure information persists across conversation threads.

4. **Vector Similarity Search**: Used the correct operator (`<->`) for cosine similarity search and implemented fallbacks for robustness.

5. **Message Serialization**: Created custom serialization/deserialization for LangGraph message objects to ensure proper state persistence.

## Future Improvements

Potential enhancements for the memory agent:

1. **Enhanced Information Extraction**: Implement NLP techniques for better entity extraction
2. **RAG Integration**: Add document retrieval capabilities using the same vector storage
3. **Memory Decay**: Implement importance-based memory decay for more human-like recall
4. **Multi-User Support**: Enhance multi-user capabilities with role-based permissions
5. **Web Interface**: Add a web UI for better visualization of conversations
6. **Conversation Summarization**: Automatically generate summaries of past conversations
7. **Emotion Recognition**: Track sentiment and emotional context across conversations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.