import os
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

# Load environment variables from .env
load_dotenv()

# ─────────────────────────────────────────────────────────────
# PostgreSQL Configuration (Configurable via Environment Variables)
# ─────────────────────────────────────────────────────────────
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "exofinder")
DB_PATH = None  # Compatibility alias

# Properly URL-encode username and password for special characters (e.g. @, #, $, etc.)
encoded_user = urllib.parse.quote_plus(POSTGRES_USER)
encoded_pass = urllib.parse.quote_plus(POSTGRES_PASSWORD)

DEFAULT_PG_URL = f"postgresql+psycopg2://{encoded_user}:{encoded_pass}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_PG_URL)

# Ensure explicit postgresql+psycopg2 driver dialect
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

IS_POSTGRES = DATABASE_URL.startswith("postgresql")

# ─────────────────────────────────────────────────────────────
# High-Concurrency Engine Setup with Connection Pooling
# ─────────────────────────────────────────────────────────────
if IS_POSTGRES:
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_pre_ping=True,      # Tests connection liveness before checkout
        pool_recycle=1800        # Recycles stale connections after 30 minutes
    )
else:
    # Fallback to local SQLite if specified
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from contextlib import contextmanager

def get_db():
    """FastAPI & script dependency yielding a clean database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    """Context manager for safe transactional operations with auto-rollback on exception."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def ensure_database_exists():
    """Connects to default PostgreSQL server and creates 'exofinder' DB if missing."""
    if not IS_POSTGRES:
        return True

    from psycopg2 import connect
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    try:
        # Connect to maintenance database 'postgres' to check/create exofinder
        conn = connect(
            dbname="postgres",
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (POSTGRES_DB,))
        exists = cur.fetchone()
        
        if not exists:
            print(f"[ExoFinder DBMS] Database '{POSTGRES_DB}' does not exist. Creating...")
            cur.execute(f'CREATE DATABASE "{POSTGRES_DB}" ENCODING \'UTF8\'')
            print(f"[ExoFinder DBMS] Database '{POSTGRES_DB}' created successfully.")
        else:
            print(f"[ExoFinder DBMS] Database '{POSTGRES_DB}' already exists.")
            
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[ExoFinder DBMS] Error checking/creating PostgreSQL database: {e}")
        return False

def check_connection():
    """Validates connectivity to PostgreSQL."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, "Connected successfully"
    except Exception as e:
        return False, str(e)

def init_db():
    """Initializes all database tables defined in models.py."""
    from .models import Target, BLSSignal, PhysicalFeature, ModelInference, UserAnnotation, PipelineJob
    Base.metadata.create_all(bind=engine)
