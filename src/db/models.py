import datetime
from sqlalchemy import (
    Column, BigInteger, String, Float, Boolean, Text, DateTime,
    ForeignKey, Index
)
from sqlalchemy.orm import relationship
from .database import Base

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

class Target(Base):
    """Core astronomical star target catalog entry."""
    __tablename__ = "targets"

    star_id = Column(BigInteger, primary_key=True, index=True, autoincrement=False)
    tic_id = Column(String(32), index=True, nullable=False)
    ground_truth = Column(String(32), index=True, nullable=False)
    catalog_period = Column(Float, nullable=True)
    raw_filepath = Column(String(255), nullable=True)
    detrended_filepath = Column(String(255), nullable=True)
    has_detrended = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # 1:1 Relationships
    bls_signal = relationship("BLSSignal", back_populates="target", uselist=False, cascade="all, delete-orphan")
    features = relationship("PhysicalFeature", back_populates="target", uselist=False, cascade="all, delete-orphan")
    inference = relationship("ModelInference", back_populates="target", uselist=False, cascade="all, delete-orphan")
    annotation = relationship("UserAnnotation", back_populates="target", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Target TIC {self.star_id} ({self.ground_truth})>"


class BLSSignal(Base):
    """Box Least Squares (BLS) periodogram search results."""
    __tablename__ = "bls_signals"

    star_id = Column(BigInteger, ForeignKey("targets.star_id", ondelete="CASCADE"), primary_key=True)
    recovered_period = Column(Float, nullable=False)
    recovered_epoch = Column(Float, default=0.0, nullable=False)
    bls_depth = Column(Float, default=0.0, nullable=False)
    bls_duration = Column(Float, default=0.0, nullable=False)
    bls_power = Column(Float, default=0.0, nullable=False)
    transit_snr = Column(Float, default=0.0, index=True, nullable=False)
    harmonic_flag = Column(String(16), default="fundamental", nullable=False)
    candidate_id = Column(String(16), default="01", nullable=False)

    target = relationship("Target", back_populates="bls_signal")


class PhysicalFeature(Base):
    """14 pure astronomical physical observables (zero data leakage) + extensible JSON metrics."""
    __tablename__ = "physical_features"

    star_id = Column(BigInteger, ForeignKey("targets.star_id", ondelete="CASCADE"), primary_key=True)
    transit_duty_cycle = Column(Float, default=0.0, index=True, nullable=False)
    mean_to_max_depth_ratio = Column(Float, default=0.0, nullable=False)
    in_transit_skew = Column(Float, default=0.0, nullable=False)
    in_transit_kurtosis = Column(Float, default=0.0, nullable=False)
    flux_kurtosis = Column(Float, default=0.0, nullable=False)
    ingress_egress_symmetry = Column(Float, default=0.0, nullable=False)
    secondary_eclipse_check = Column(Float, default=0.0, nullable=False)
    depth_to_noise = Column(Float, default=0.0, nullable=False)
    odd_even_depth_ratio = Column(Float, default=1.0, nullable=False)
    odd_even_mismatch_sigma = Column(Float, default=0.0, nullable=False)
    std_raw = Column(Float, default=0.0, nullable=False)
    std_detrended = Column(Float, default=0.0, nullable=False)
    transit_count = Column(Float, default=0.0, nullable=False)
    extra_metrics_json = Column(Text, nullable=True)

    target = relationship("Target", back_populates="features")


class ModelInference(Base):
    """Predictions, probability distributions, provenance, and consensus verdicts from ML & 1D-CNN."""
    __tablename__ = "model_inferences"

    star_id = Column(BigInteger, ForeignKey("targets.star_id", ondelete="CASCADE"), primary_key=True)
    model_version = Column(String(32), default="v1.0", nullable=False)
    
    # Stacking ML Ensemble results
    ml_predicted_label = Column(String(32), index=True, nullable=False)
    ml_confidence = Column(Float, default=0.0, index=True, nullable=False)
    prob_ml_planet = Column(Float, default=0.0, nullable=False)
    prob_ml_binary = Column(Float, default=0.0, nullable=False)
    prob_ml_fp = Column(Float, default=0.0, nullable=False)
    significance = Column(String(16), default="high", nullable=False)

    # NASA Astronet 1D-CNN results
    cnn_predicted_label = Column(String(32), index=True, nullable=True)
    cnn_confidence = Column(Float, default=0.0, nullable=True)
    prob_cnn_planet = Column(Float, default=0.0, nullable=True)
    prob_cnn_binary = Column(Float, default=0.0, nullable=True)
    prob_cnn_fp = Column(Float, default=0.0, nullable=True)

    # Consensus verdict (unanimous / split / ambiguous)
    consensus_verdict = Column(String(32), default="unanimous", nullable=False)

    target = relationship("Target", back_populates="inference")


class UserAnnotation(Base):
    """User curation, bookmark flags, and astronomical vetting notes."""
    __tablename__ = "user_annotations"

    star_id = Column(BigInteger, ForeignKey("targets.star_id", ondelete="CASCADE"), primary_key=True)
    is_bookmarked = Column(Boolean, default=False, index=True, nullable=False)
    user_notes = Column(Text, default="", nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    target = relationship("Target", back_populates="annotation")


class PipelineJob(Base):
    """Tracks on-demand user analysis jobs for any star."""
    __tablename__ = "pipeline_jobs"

    job_id = Column(String(36), primary_key=True, index=True)
    tic_id = Column(String(32), index=True, nullable=False)
    star_id = Column(BigInteger, nullable=True, index=True)
    status = Column(String(32), default="QUEUED", index=True, nullable=False)
    current_stage = Column(String(64), default="Queued in processing engine", nullable=False)
    progress_pct = Column(Float, default=0.0, nullable=False)
    error_message = Column(Text, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<PipelineJob {self.job_id[:8]} TIC {self.tic_id} ({self.status})>"


# High-Performance Compound Query Indices
Index("idx_filter_ml_pred_conf", ModelInference.ml_predicted_label, ModelInference.ml_confidence.desc())
Index("idx_bls_snr_desc", BLSSignal.transit_snr.desc())
Index("idx_target_lookup", Target.star_id, Target.ground_truth)
Index("idx_user_annotations_bookmarked", UserAnnotation.is_bookmarked, UserAnnotation.updated_at.desc())
Index("idx_pipeline_jobs_created", PipelineJob.created_at.desc())

