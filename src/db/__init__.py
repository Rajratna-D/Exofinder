"""
ExoFinder Nova Database Package
High-concurrency SQLite WAL Engine & ORM Models
"""
from .database import engine, SessionLocal, get_db, get_db_session, init_db, DATABASE_URL, DB_PATH
from .models import Base, Target, BLSSignal, PhysicalFeature, ModelInference, UserAnnotation, PipelineJob
from .repository import (
    get_targets_page,
    get_target_details,
    toggle_bookmark,
    save_target_notes,
    create_pipeline_job,
    update_pipeline_job,
    get_pipeline_job
)

__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_session",
    "init_db",
    "DATABASE_URL",
    "DB_PATH",
    "Base",
    "Target",
    "BLSSignal",
    "PhysicalFeature",
    "ModelInference",
    "UserAnnotation",
    "PipelineJob",
    "get_targets_page",
    "get_target_details",
    "toggle_bookmark",
    "save_target_notes",
    "create_pipeline_job",
    "update_pipeline_job",
    "get_pipeline_job",
]

