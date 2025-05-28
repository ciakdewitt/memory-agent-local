#!/usr/bin/env python3
"""
Schema fixer for the memory agent.
Ensures that the PostgreSQL database has the correct schema for the memory agent.
"""

import os
from dotenv import load_dotenv
import psycopg
import time

# Load environment variables
load_dotenv()

# Database connection parameters
db_url = os.getenv("DATABASE_URL", "postgresql://postgres:mgkcharlie@localhost:5432/memory_agent")

def fix_schema():
    """Fix the database schema."""
    print("Fixing database schema...")
    
    # Try to connect with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Connect to the database
            conn = psycopg.connect(db_url, autocommit=True)
            
            with conn.cursor() as cur:
                # 1. Ensure pgvector extension exists
                print("Checking pgvector extension...")
                cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
                result = cur.fetchone()
                
                if result[0] == 0:
                    print("Creating pgvector extension...")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                else:
                    print("pgvector extension already exists")
                
                # 2. Fix message_vectors table
                print("Creating/updating message_vectors table...")
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
                
                # 3. Create indices for vector search
                print("Setting up vector indices...")
                # First check if the index exists
                cur.execute("""
                    SELECT COUNT(*) FROM pg_indexes 
                    WHERE indexname = 'message_vectors_embedding_idx'
                """)
                
                if cur.fetchone()[0] == 0:
                    print("Creating vector index...")
                    # Try different index types with fallbacks
                    index_types = [
                        "CREATE INDEX message_vectors_embedding_idx ON message_vectors USING hnsw (embedding vector_cosine_ops)",
                        "CREATE INDEX message_vectors_embedding_idx ON message_vectors USING ivfflat (embedding vector_cosine_ops)",
                        "CREATE INDEX message_vectors_embedding_idx ON message_vectors USING ivfflat (embedding vector_l2_ops)"
                    ]
                    
                    success = False
                    for index_query in index_types:
                        try:
                            cur.execute(index_query)
                            print(f"Successfully created index with: {index_query}")
                            success = True
                            break
                        except Exception as e:
                            print(f"Failed to create index with {index_query.split()[3]}: {str(e)}")
                    
                    if not success:
                        print("WARNING: Could not create any vector index. Vector search may be slow.")
                else:
                    print("Vector index already exists")
                
                # 4. Create user_threads table
                print("Creating user_threads table...")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_threads (
                        user_id TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 5. Check LangGraph checkpoint table
                print("Checking LangGraph tables...")
                try:
                    cur.execute("SELECT 1 FROM langgraph_checkpoints LIMIT 1")
                    print("LangGraph tables already set up")
                except Exception:
                    print("Setting up LangGraph checkpoint table...")
                    # Create the table LangGraph needs
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                            id TEXT PRIMARY KEY,
                            config JSONB,
                            values JSONB,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                
                print("Schema fixed successfully")
                conn.close()
                return True
                
        except Exception as e:
            print(f"Error fixing schema (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {2**attempt} seconds...")
                time.sleep(2**attempt)  # Exponential backoff
            else:
                print("All attempts failed. Database setup incomplete.")
                return False
        finally:
            try:
                if 'conn' in locals() and conn:
                    conn.close()
            except:
                pass

if __name__ == "__main__":
    fix_schema()