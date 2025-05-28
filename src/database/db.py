"""
Database connection and setup for the Memory Agent with pgvector support.
"""

import os
import time
from typing import Optional
from contextlib import contextmanager
import psycopg

def get_db_connection(max_retries=3, retry_delay=1):
    """
    Get a PostgreSQL connection with pgvector support.
    """
    # Database connection string
    db_uri = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:mgkcharlie@localhost:5432/memory_agent"
    )
    
    print(f"Connecting to PostgreSQL database...")
    
    # Try to connect with retries
    retries = 0
    last_error = None
    
    while retries < max_retries:
        try:
            # Connect with autocommit disabled and no pooling
            conn = psycopg.connect(
                db_uri,
                autocommit=False,
                prepare_threshold=0  # Disable prepared statements
            )
            
            # Verify pgvector extension is available
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
                result = cur.fetchone()
                has_vector = result[0] > 0
                
                if not has_vector:
                    print("Warning: pgvector extension not found")
                    print("Attempting to create pgvector extension...")
                    try:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        conn.commit()
                        print("Successfully created pgvector extension")
                    except Exception as e:
                        conn.rollback()
                        print(f"Error creating pgvector extension: {str(e)}")
                        print("Please run fix_schema.py to set up the database properly")
                else:
                    print("pgvector extension is available")
            
            print("Successfully connected to PostgreSQL")
            return conn
            
        except Exception as e:
            last_error = str(e)
            retries += 1
            
            if retries < max_retries:
                print(f"Connection attempt {retries} failed: {last_error}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect after {max_retries} attempts: {last_error}")
    
    # If we reach here, all connection attempts failed
    raise ConnectionError(f"Could not connect to PostgreSQL: {last_error}")

@contextmanager
def db_cursor(dict_results=False):
    """
    Context manager for database operations with automatic connection handling.
    
    Args:
        dict_results: If True, cursor will return dictionaries. Set to False for LangGraph compatibility.
    """
    conn = None
    try:
        conn = get_db_connection()
        
        if dict_results:
            # For human-readable results, use dict_row
            from psycopg.rows import dict_row
            with conn.cursor(row_factory=dict_row) as cur:
                yield cur
        else:
            # For LangGraph compatibility, use default tuple results
            with conn.cursor() as cur:
                yield cur
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # When run directly, test connection
    print("Testing database connection...")
    try:
        conn = get_db_connection()
        print("Database connection successful!")
        
        # List tables
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cur.fetchall()]
            print(f"Tables in database: {tables}")
        
        print("Connection test completed successfully!")
    except Exception as e:
        print(f"Connection test failed: {str(e)}")