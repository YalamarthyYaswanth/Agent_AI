# tools/sql_tool.py — POSTGRESQL + PUBLIC SCHEMA
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT", "5432")
    )
    return conn

def execute_sql_query(query: str) -> str:
    """Execute SQL query and return formatted results."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        
        if cursor.description is None:
            return "Query executed but returned no data (e.g., INSERT/UPDATE)."

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        if not rows:
            return "Query returned no data."

        from tabulate import tabulate
        table = tabulate(rows, headers=columns, tablefmt="grid")
        return f"Results:\n{table}"

    except Exception as e:
        return f"SQL Error: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()

def get_table_schema(table_name: str) -> str:
    """Get column names and types for a PostgreSQL table."""
    if ";" in table_name or "--" in table_name:
        return "Invalid table name."

    # PostgreSQL folds unquoted names to lowercase
    query = f"""
    SELECT 
        column_name, 
        data_type 
    FROM information_schema.columns 
    WHERE table_name = '{table_name.lower()}'
    ORDER BY ordinal_position;
    """
    return execute_sql_query(query)

def list_tables(dummy: str = "") -> str:
    """List all tables in 'public' schema."""
    query = """
    SELECT tablename 
    FROM pg_tables 
    WHERE schemaname = 'public';
    """
    return execute_sql_query(query)