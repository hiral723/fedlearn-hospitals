"""
SQLAlchemy models for the FedHealth app.

Hospital       -- one of the 20 simulated hospitals (FL participant nodes)
User           -- a login: role is 'admin' (FL coordinator), 'hospital'
                  (FL participant, tied to a Hospital), or 'patient'
                  (clinical data subject -- not an FL participant)
PredictionLog  -- lightweight log of hospital-side classify actions
TrainingRun    -- a record of a live federated training session
PatientRecord  -- a patient's uploaded X-ray + optional report, with the
                  stored files, prediction, and OCR'd report text
"""

from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class Hospital(db.Model):
    __tablename__ = "hospitals"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    cluster_id = db.Column(db.Integer, nullable=True)
    total_images = db.Column(db.Integer, default=0)

    users = db.relationship("User", backref="hospital", lazy=True)
    predictions = db.relationship("PredictionLog", backref="hospital", lazy=True)


class User(db.Model, UserMixin):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)        # 'admin' | 'hospital' | 'patient'
    hospital_id = db.Column(db.Integer, db.ForeignKey("hospitals.id"), nullable=True)
    display_name = db.Column(db.String(120), nullable=True)  # patients' full name
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def is_admin(self):
        return self.role == "admin"

    def is_hospital(self):
        return self.role == "hospital"

    def is_patient(self):
        return self.role == "patient"


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"
    id = db.Column(db.Integer, primary_key=True)
    hospital_id = db.Column(db.Integer, db.ForeignKey("hospitals.id"), nullable=True)
    predicted_class = db.Column(db.String(40), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    source_filename = db.Column(db.String(255), nullable=True)  # set for batch zip uploads
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))


class TrainingRun(db.Model):
    __tablename__ = "training_runs"
    id = db.Column(db.Integer, primary_key=True)
    hospital_ids = db.Column(db.String(200), nullable=False)
    num_rounds = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default="running")
    final_accuracy = db.Column(db.Float, nullable=True)
    started_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    finished_at = db.Column(db.DateTime, nullable=True)
    triggered_by = db.Column(db.String(80), nullable=True)


class PatientRecord(db.Model):
    __tablename__ = "patient_records"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    xray_filename = db.Column(db.String(255), nullable=True)
    predicted_class = db.Column(db.String(40), nullable=True)
    confidence = db.Column(db.Float, nullable=True)

    report_filename = db.Column(db.String(255), nullable=True)
    report_text = db.Column(db.Text, nullable=True)
    report_keywords = db.Column(db.String(255), nullable=True)  # comma-separated matches

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    patient = db.relationship("User", backref="records")
