"""
Creates the database and populates it with:
  - one Hospital row per simulated hospital
  - one login account per hospital (username: hospital0, hospital1, ... /
    password: same as username)
  - one admin account (username: admin / password: admin123)

Patient accounts are NOT seeded here -- patients self-register via /register.

Run this ONCE, from project root, after partition.py and cluster_clients.py:
    python webapp/backend/seed_db.py

Safe to re-run: it wipes and rebuilds the database each time (patient
accounts and their records will also be wiped on re-run).
"""

import json
import sys
from pathlib import Path

from werkzeug.security import generate_password_hash

sys.path.append(str(Path(__file__).parent))
from app import app  # noqa: E402
from models import db, Hospital, User  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLIENT_INDICES_PATH = PROJECT_ROOT / "simulation" / "client_indices.json"
CLUSTER_ASSIGNMENTS_PATH = PROJECT_ROOT / "clustering" / "cluster_assignments.json"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


def main():
    if not CLIENT_INDICES_PATH.exists():
        raise SystemExit(f"{CLIENT_INDICES_PATH} not found. Run simulation/partition.py first.")

    with open(CLIENT_INDICES_PATH) as f:
        client_samples = json.load(f)

    cluster_of = {}
    if CLUSTER_ASSIGNMENTS_PATH.exists():
        with open(CLUSTER_ASSIGNMENTS_PATH) as f:
            clusters = json.load(f)
        for cid, hospital_ids in clusters.items():
            for hid in hospital_ids:
                cluster_of[hid] = int(cid)

    with app.app_context():
        db.drop_all()
        db.create_all()

        for hid, samples in enumerate(client_samples):
            hospital = Hospital(
                id=hid, name=f"Hospital {hid}",
                cluster_id=cluster_of.get(hid), total_images=len(samples),
            )
            db.session.add(hospital)

            user = User(
                username=f"hospital{hid}",
                password_hash=generate_password_hash(f"hospital{hid}"),
                role="hospital", hospital_id=hid, display_name=f"Hospital {hid}",
            )
            db.session.add(user)

        admin = User(
            username=ADMIN_USERNAME,
            password_hash=generate_password_hash(ADMIN_PASSWORD),
            role="admin", hospital_id=None, display_name="Administrator",
        )
        db.session.add(admin)

        db.session.commit()

        print(f"Seeded {len(client_samples)} hospitals + logins.")
        print(f"Hospital logins: hospital0/hospital0 through hospital{len(client_samples)-1}/hospital{len(client_samples)-1}")
        print(f"Admin login: {ADMIN_USERNAME}/{ADMIN_PASSWORD}")
        print("Patient accounts are created via self-registration at /register.")


if __name__ == "__main__":
    main()
