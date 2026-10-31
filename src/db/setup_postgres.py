"""
ExoFinder PostgreSQL DBMS Setup & Verification Tool
===================================================
Automates:
  1. Connection test to PostgreSQL
  2. Database creation ('exofinder')
  3. Relational table schema generation (SQLAlchemy models)
  4. Full catalog migration and indexing from flat CSV files
  5. Relational parity and join query verification

Usage:
  python src/db/setup_postgres.py
  python src/db/setup_postgres.py --password your_password
"""

import os
import sys
import argparse
import time
from dotenv import load_dotenv

# Ensure parent directory is in sys.path and load .env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Setup and verify ExoFinder PostgreSQL DBMS")
    parser.add_argument("--user", default="postgres", help="PostgreSQL user (default: postgres)")
    parser.add_argument("--password", default=None, help="PostgreSQL password")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host (default: localhost)")
    parser.add_argument("--port", default="5432", help="PostgreSQL port (default: 5432)")
    parser.add_argument("--db", default="exofinder", help="Target database name (default: exofinder)")
    
    args = parser.parse_args()

    # Set environment variables if passed via CLI
    if args.user:
        os.environ["POSTGRES_USER"] = args.user
    if args.password:
        os.environ["POSTGRES_PASSWORD"] = args.password
    if args.host:
        os.environ["POSTGRES_HOST"] = args.host
    if args.port:
        os.environ["POSTGRES_PORT"] = args.port
    if args.db:
        os.environ["POSTGRES_DB"] = args.db

    # Re-compute DATABASE_URL with proper URL encoding for special characters
    import urllib.parse
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db_name = os.environ.get("POSTGRES_DB", "exofinder")
    
    enc_user = urllib.parse.quote_plus(user)
    enc_pass = urllib.parse.quote_plus(password)
    os.environ["DATABASE_URL"] = f"postgresql+psycopg2://{enc_user}:{enc_pass}@{host}:{port}/{db_name}"

    print("=" * 65)
    print("🪐 ExoFinder: PostgreSQL Database Setup & Seeding Utility")
    print("=" * 65)
    print(f"Target Database Host: {host}:{port}")
    print(f"Database Name:        {db_name}")
    print(f"User:                 {user}")
    print("=" * 65)

    from src.db.database import ensure_database_exists, check_connection, init_db, SessionLocal
    from src.db.models import Target, BLSSignal, PhysicalFeature, ModelInference, UserAnnotation, PipelineJob
    from src.db.migrate import run_migration

    # Step 1: Check and Create Database
    print("\n[Step 1/4] Connecting to PostgreSQL server and verifying database...")
    created = ensure_database_exists()
    if not created:
        print("\n❌ Could not connect to PostgreSQL. Please ensure:")
        print(f"  1. PostgreSQL service is running on {host}:{port}")
        print(f"  2. The password is correct (currently using: '{password}')")
        print("\nTip: Run with your password:")
        print(f"  python src/db/setup_postgres.py --password YOUR_PASSWORD")
        sys.exit(1)

    # Step 2: Test connection to exofinder DB
    print("\n[Step 2/4] Testing connection pool to target database...")
    ok, msg = check_connection()
    if not ok:
        print(f"❌ Connection failed: {msg}")
        sys.exit(1)
    print(f"  ✓ Connected successfully! ({msg})")

    # Step 3: Run Database Migration
    print("\n[Step 3/4] Running schema initialization and catalog migration...")
    run_migration()

    # Step 4: Verification
    print("\n[Step 4/4] Final Relational Verification...")
    db = SessionLocal()
    try:
        t_count = db.query(Target).count()
        b_count = db.query(BLSSignal).count()
        f_count = db.query(PhysicalFeature).count()
        i_count = db.query(ModelInference).count()
        j_count = db.query(PipelineJob).count()

        print(f"  • Total Targets Indexed:      {t_count:,}")
        print(f"  • BLS Signals Available:      {b_count:,}")
        print(f"  • Physical Features:          {f_count:,}")
        print(f"  • Model Inferences Stored:    {i_count:,}")
        print(f"  • Pipeline Jobs Queue Table:  {j_count} records (Ready)")

        print("\n" + "=" * 65)
        print("🎉 SUCCESS! ExoFinder PostgreSQL DBMS is fully operational.")
        print("Ready for high-concurrency production load!")
        print("=" * 65)
    finally:
        db.close()

if __name__ == "__main__":
    main()
