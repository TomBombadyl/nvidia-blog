#!/usr/bin/env python3
"""
Script to run BigQuery migration SQL.
"""

import sys
import os
from google.cloud import bigquery

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_migration():
    """Run the BigQuery migration SQL."""
    # Read SQL file
    sql_file = os.path.join(os.path.dirname(__file__), '..', 'bigquery', 'migrations', '001_bigquery_setup.sql')
    
    with open(sql_file, 'r') as f:
        sql_content = f.read()
    
    # Initialize BigQuery client
    client = bigquery.Client(project='nvidia-blog')
    
    # Split SQL into individual statements
    # Remove comments and split by semicolon, but keep multi-line statements together
    statements = []
    current_statement = []
    
    for line in sql_content.split('\n'):
        stripped = line.strip()
        # Skip empty lines and comments
        if not stripped or stripped.startswith('--'):
            continue
        current_statement.append(line)
        if stripped.endswith(';'):
            statements.append('\n'.join(current_statement))
            current_statement = []
    
    if current_statement:
        statements.append('\n'.join(current_statement))
    
    print("Executing BigQuery migration...")
    print(f"SQL file: {sql_file}")
    print(f"Found {len(statements)} SQL statements to execute\n")
    
    try:
        # Execute each statement separately
        for i, sql in enumerate(statements, 1):
            if not sql.strip():
                continue
            print(f"Executing statement {i}/{len(statements)}...")
            print(f"SQL: {sql[:100]}...")
            
            query_job = client.query(sql, location='europe-west3')
            results = query_job.result()  # Wait for completion
            print(f"  [SUCCESS] Statement {i} completed (Job ID: {query_job.job_id})")
        
        print("\n[SUCCESS] Migration completed successfully!")
        print(f"Job ID: {query_job.job_id}")
        
        # Verify tables were created
        dataset_ref = client.dataset('nvidia_blog', project='nvidia-blog')
        try:
            dataset = client.get_dataset(dataset_ref)
            print(f"\n[SUCCESS] Dataset '{dataset.dataset_id}' exists")
            
            # List tables
            tables = list(client.list_tables(dataset_ref))
            print(f"\n[SUCCESS] Found {len(tables)} tables:")
            for table in tables:
                print(f"   - {table.table_id}")
                
        except Exception as e:
            print(f"\n[WARNING] Could not verify dataset: {e}")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Error executing migration: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_migration())

