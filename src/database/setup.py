"""
Database setup script for the Memory Agent.
"""

from psycopg import Connection

def setup_database():
    """Set up the database tables needed for the Memory Agent."""
    try:
        # Direct connection parameters
        conn = Connection.connect(
            "postgresql://postgres:example@localhost:5432/memory_agent?sslmode=disable",
            autocommit=True,
            prepare_threshold=0
        )
        
        with conn.cursor() as cur:
            # Create user_threads table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_threads (
                    user_id TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            print("Database tables created successfully")
            
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        print("Connection details:", "postgres:example@localhost:5432/memory_agent")

if __name__ == "__main__":
    setup_database()