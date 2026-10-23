"""
Flask backend for the FedHealth app.

Three roles:
  admin    -- FL coordinator: triggers live training, sees the hospital network
  hospital -- FL participant: classifies X-rays, can batch-classify a zip
              of images, sees only its own prediction history
  patient  -- clinical data subject (not an FL participant): uploads their
              own X-ray + optional report, sees only their own records

Pages:
  GET  /, /login, /register, POST /login, POST /register, GET /logout

API (auth):
  GET  /api/me, POST /api/login, POST /api/register

API (classify / hospital & admin):
  POST /api/predict        -> classify an X-ray, logged to PredictionLog
  POST /api/batch-classify -> (hospital/admin) upload a .zip of images,
                               classify every image inside, return a table
  GET  /api/my-history     -> (hospital) this hospital's past predictions
  GET  /api/hospitals      -> all hospitals + cluster membership (from DB)

API (patient):
  POST /api/patient/upload-xray   -> classify + save a patient's X-ray
  POST /api/patient/upload-report -> OCR a patient's report, optionally
                                      attached to an existing record
  GET  /api/patient/my-records    -> this patient's past records
  GET  /api/patient/record/<id>   -> full detail (regenerates heatmap)

API (research / shared):
  GET  /api/metrics, /api/architecture, /api/results

API (admin / live training):
  GET  /api/training-runs, POST /api/live-train, POST /api/stop-training

Files:
  GET /uploads/<kind>/<filename> -> serves a stored X-ray/report (owner-checked)

Socket.IO events: 'training_progress', 'training_complete', 'training_error'

Run from project root:
    python webapp/backend/app.py
Then open http://127.0.0.1:5000
"""

import base64
import io
import json
import os
import sys
import uuid
import zipfile
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, redirect, url_for, abort
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from flask_socketio import SocketIO
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

SIM_DIR = str(Path(__file__).parent.parent.parent / "simulation")
sys.path.append(SIM_DIR)
from model import ChestCNN, CLASSES, TRANSFORM, get_device  # noqa: E402
from gradcam import GradCAM, overlay_heatmap, get_bounding_box, scale_box_to_image, draw_box  # noqa: E402
from models import db, Hospital, User, PredictionLog, TrainingRun, PatientRecord  # noqa: E402
import live_trainer  # noqa: E402
import ocr  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
MODEL_PATH = Path(__file__).parent / "demo_model.pt"
METRICS_PATH = Path(__file__).parent / "metrics.json"
DB_PATH = Path(__file__).parent / "fedhealth.db"
UPLOADS_DIR = Path(__file__).parent / "uploads"
XRAY_UPLOAD_DIR = UPLOADS_DIR / "xrays"
REPORT_UPLOAD_DIR = UPLOADS_DIR / "reports"
BASELINE_RESULTS_PATH = PROJECT_ROOT / "simulation" / "results_baseline.json"
CLUSTERED_RESULTS_PATH = PROJECT_ROOT / "clustering" / "results_clustered.json"

XRAY_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}

DEVICE = get_device()

app = Flask(__name__, static_folder=None)
app.config["SECRET_KEY"] = "fedhealth-dev-secret-change-if-deploying"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB, generous for dataset zips

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = "login_page"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# --- load the trained model once at startup ---
model = None
gradcam = None
model_load_error = None
if MODEL_PATH.exists():
    try:
        model = ChestCNN().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        gradcam = GradCAM(model, model.features[-1])
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        model_load_error = str(e)
        print(f"ERROR loading model: {e}")
else:
    model_load_error = f"No model found at {MODEL_PATH}. Run train_demo_model.py first."
    print(model_load_error)


def allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT


def classify_image_bytes(img_bytes, with_heatmaps=False):
    """Core inference used by /api/predict, patient uploads, and batch classify."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((128, 128))
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_idx = int(probs.argmax())

    result = {
        "predicted_class": CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
    }

    if with_heatmaps:
        heatmaps = {}
        predicted_box = None
        try:
            for class_idx, class_name in enumerate(CLASSES):
                x_grad = TRANSFORM(img).unsqueeze(0).to(DEVICE)
                x_grad.requires_grad_(True)
                cam = gradcam.generate(x_grad, class_idx)
                overlay = overlay_heatmap(img, cam)
                if class_idx == predicted_idx:
                    box = get_bounding_box(cam, threshold_frac=0.5)
                    if box is not None:
                        pixel_box = scale_box_to_image(box, cam.shape, img.size)
                        overlay = draw_box(overlay, pixel_box, color=(255, 255, 255), width=3)
                        predicted_box = {"x_min": pixel_box[0], "y_min": pixel_box[1], "x_max": pixel_box[2], "y_max": pixel_box[3]}
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                heatmaps[class_name] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Grad-CAM generation failed (non-fatal): {e}")
        result["heatmaps"] = heatmaps
        result["bounding_box"] = predicted_box

    return result


# ===================== Pages =====================
@app.route("/")
def serve_index():
    if not current_user.is_authenticated:
        return redirect(url_for("login_page"))
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/login", methods=["GET"])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for("serve_index"))
    return send_from_directory(FRONTEND_DIR, "login.html")


@app.route("/register", methods=["GET"])
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for("serve_index"))
    return send_from_directory(FRONTEND_DIR, "register.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    user = User.query.filter_by(username=username).first()
    if user is None or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid username or password"}), 401

    login_user(user)
    return jsonify({"ok": True, "role": user.role})


@app.route("/api/register", methods=["POST"])
def api_register():
    """Self-signup -- PATIENTS ONLY. Hospital and admin accounts are
    pre-seeded via seed_db.py and never created through this endpoint."""
    data = request.get_json(force=True)
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    display_name = (data.get("display_name") or "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken"}), 409

    user = User(
        username=username,
        password_hash=generate_password_hash(password),
        role="patient",
        display_name=display_name or username,
    )
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify({"ok": True, "role": "patient"})


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("login_page"))


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ===================== Auth info =====================
@app.route("/api/me")
@login_required
def me():
    return jsonify({
        "username": current_user.username,
        "role": current_user.role,
        "hospital_id": current_user.hospital_id,
        "display_name": current_user.display_name,
    })


# ===================== Predict (hospital/admin direct classify) =====================
@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        return jsonify({"error": model_load_error}), 503
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    try:
        img_bytes = file.read()
        result = classify_image_bytes(img_bytes, with_heatmaps=True)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 400

    log = PredictionLog(
        hospital_id=current_user.hospital_id,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
    )
    db.session.add(log)
    db.session.commit()

    return jsonify(result)


@app.route("/api/my-history")
@login_required
def my_history():
    if current_user.hospital_id is None:
        return jsonify({"error": "Admin accounts have no hospital history"}), 400
    logs = (
        PredictionLog.query.filter_by(hospital_id=current_user.hospital_id)
        .order_by(PredictionLog.timestamp.desc())
        .limit(50)
        .all()
    )
    return jsonify({
        "history": [
            {"predicted_class": log.predicted_class, "confidence": log.confidence,
             "timestamp": log.timestamp.isoformat(), "source_filename": log.source_filename}
            for log in logs
        ]
    })


# ===================== Batch classify (hospital/admin, zip of images) =====================
@app.route("/api/batch-classify", methods=["POST"])
@login_required
def batch_classify():
    if not (current_user.is_hospital() or current_user.is_admin()):
        return jsonify({"error": "Hospital or admin accounts only"}), 403
    if model is None:
        return jsonify({"error": model_load_error}), 503
    if "dataset" not in request.files:
        return jsonify({"error": "No zip file uploaded (field name 'dataset')"}), 400

    zip_file = request.files["dataset"]
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        zip_path = tmp_dir_path / "upload.zip"
        zip_file.save(zip_path)

        try:
            with zipfile.ZipFile(zip_path) as zf:
                # zip-slip safe extraction: resolve each member and verify
                # it stays inside tmp_dir_path before extracting
                for member in zf.namelist():
                    member_path = (tmp_dir_path / member).resolve()
                    if not str(member_path).startswith(str(tmp_dir_path.resolve())):
                        continue  # skip anything trying to escape the extraction dir
                zf.extractall(tmp_dir_path)
        except zipfile.BadZipFile:
            return jsonify({"error": "That doesn't look like a valid .zip file"}), 400

        image_files = [
            p for p in tmp_dir_path.rglob("*")
            if p.is_file() and p.suffix.lower().lstrip(".") in ALLOWED_IMAGE_EXT and p.name != "upload.zip"
        ]

        if not image_files:
            return jsonify({"error": "No image files (.png/.jpg/.jpeg) found inside the zip"}), 400

        for img_path in sorted(image_files):
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                result = classify_image_bytes(img_bytes, with_heatmaps=False)
                results.append({
                    "filename": img_path.name,
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                })
                log = PredictionLog(
                    hospital_id=current_user.hospital_id,
                    predicted_class=result["predicted_class"],
                    confidence=result["confidence"],
                    source_filename=img_path.name,
                )
                db.session.add(log)
            except Exception as e:
                results.append({"filename": img_path.name, "error": str(e)})

        db.session.commit()

    class_counts = {}
    for r in results:
        if "predicted_class" in r:
            class_counts[r["predicted_class"]] = class_counts.get(r["predicted_class"], 0) + 1

    return jsonify({"results": results, "total": len(results), "class_counts": class_counts})


# ===================== Hospitals (from DB) =====================
@app.route("/api/hospitals")
@login_required
def hospitals():
    all_hospitals = Hospital.query.order_by(Hospital.id).all()
    return jsonify({
        "hospitals": [
            {
                "id": h.id, "name": h.name, "cluster": h.cluster_id,
                "total_images": h.total_images, "prediction_count": len(h.predictions),
            }
            for h in all_hospitals
        ]
    })


# ===================== Patient: upload X-ray =====================
@app.route("/api/patient/upload-xray", methods=["POST"])
@login_required
def patient_upload_xray():
    if not current_user.is_patient():
        return jsonify({"error": "Patient accounts only"}), 403
    if model is None:
        return jsonify({"error": model_load_error}), 503
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files["image"]
    if not allowed_image(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    img_bytes = file.read()
    try:
        result = classify_image_bytes(img_bytes, with_heatmaps=True)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 400

    ext = secure_filename(file.filename).rsplit(".", 1)[-1]
    stored_name = f"{uuid.uuid4().hex}.{ext}"
    with open(XRAY_UPLOAD_DIR / stored_name, "wb") as f:
        f.write(img_bytes)

    record = PatientRecord(
        patient_id=current_user.id,
        xray_filename=stored_name,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
    )
    db.session.add(record)
    db.session.commit()

    result["record_id"] = record.id
    return jsonify(result)


# ===================== Patient: upload report (OCR) =====================
@app.route("/api/patient/upload-report", methods=["POST"])
@login_required
def patient_upload_report():
    if not current_user.is_patient():
        return jsonify({"error": "Patient accounts only"}), 403
    if "report" not in request.files:
        return jsonify({"error": "No report file uploaded"}), 400

    file = request.files["report"]
    if not allowed_image(file.filename):
        return jsonify({"error": "Report must be an image (png/jpg) -- scan or photo of the document"}), 400

    record_id = request.form.get("record_id")
    ext = secure_filename(file.filename).rsplit(".", 1)[-1]
    stored_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = REPORT_UPLOAD_DIR / stored_name
    file.save(save_path)

    try:
        text = ocr.extract_text(str(save_path))
    except Exception as e:
        return jsonify({"error": f"OCR failed: {e}"}), 500

    keywords = ocr.find_keywords(text)

    if record_id:
        record = db.session.get(PatientRecord, int(record_id))
        if record is None or record.patient_id != current_user.id:
            return jsonify({"error": "Record not found"}), 404
    else:
        record = PatientRecord(patient_id=current_user.id)
        db.session.add(record)

    record.report_filename = stored_name
    record.report_text = text
    record.report_keywords = ",".join(keywords)
    db.session.commit()

    return jsonify({
        "record_id": record.id,
        "report_text": text,
        "keywords_found": keywords,
    })


@app.route("/api/patient/my-records")
@login_required
def patient_my_records():
    if not current_user.is_patient():
        return jsonify({"error": "Patient accounts only"}), 403
    records = (
        PatientRecord.query.filter_by(patient_id=current_user.id)
        .order_by(PatientRecord.created_at.desc())
        .all()
    )
    return jsonify({
        "records": [
            {
                "id": r.id,
                "xray_filename": r.xray_filename,
                "predicted_class": r.predicted_class,
                "confidence": r.confidence,
                "report_filename": r.report_filename,
                "report_text": r.report_text,
                "report_keywords": r.report_keywords.split(",") if r.report_keywords else [],
                "created_at": r.created_at.isoformat(),
            }
            for r in records
        ]
    })


@app.route("/api/patient/record/<int:record_id>")
@login_required
def patient_record_detail(record_id):
    if not current_user.is_patient():
        return jsonify({"error": "Patient accounts only"}), 403
    record = db.session.get(PatientRecord, record_id)
    if record is None or record.patient_id != current_user.id:
        return jsonify({"error": "Record not found"}), 404

    data = {
        "id": record.id, "xray_filename": record.xray_filename,
        "predicted_class": record.predicted_class, "confidence": record.confidence,
        "report_filename": record.report_filename, "report_text": record.report_text,
        "report_keywords": record.report_keywords.split(",") if record.report_keywords else [],
        "created_at": record.created_at.isoformat(), "heatmaps": None,
    }

    if record.xray_filename and model is not None:
        try:
            xray_path = XRAY_UPLOAD_DIR / record.xray_filename
            with open(xray_path, "rb") as f:
                img_bytes = f.read()
            full_result = classify_image_bytes(img_bytes, with_heatmaps=True)
            data["heatmaps"] = full_result["heatmaps"]
            data["probabilities"] = full_result["probabilities"]
        except Exception as e:
            print(f"Could not regenerate heatmap for record {record_id}: {e}")

    return jsonify(data)


# ===================== File serving (owner-checked) =====================
@app.route("/uploads/<kind>/<filename>")
@login_required
def serve_upload(kind, filename):
    if kind not in ("xrays", "reports"):
        abort(404)
    base_dir = XRAY_UPLOAD_DIR if kind == "xrays" else REPORT_UPLOAD_DIR
    filename = secure_filename(filename)

    if current_user.is_patient():
        field = PatientRecord.xray_filename if kind == "xrays" else PatientRecord.report_filename
        owns_it = PatientRecord.query.filter_by(patient_id=current_user.id).filter(field == filename).first()
        if not owns_it:
            abort(403)

    return send_from_directory(base_dir, filename)


# ===================== Research-side endpoints =====================
@app.route("/api/results")
@login_required
def results():
    data = {"baseline": None, "clusters": None}
    if BASELINE_RESULTS_PATH.exists():
        with open(BASELINE_RESULTS_PATH) as f:
            data["baseline"] = json.load(f)
    if CLUSTERED_RESULTS_PATH.exists():
        with open(CLUSTERED_RESULTS_PATH) as f:
            data["clusters"] = json.load(f)
    return jsonify(data)


@app.route("/api/metrics")
@login_required
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "No metrics found. Run compute_metrics.py first."}), 404
    with open(METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/api/architecture")
@login_required
def architecture():
    if model is None:
        return jsonify({"error": model_load_error}), 503
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers.append({"name": name, "type": "Conv2D",
                            "detail": f"{module.in_channels} -> {module.out_channels} channels, {module.kernel_size[0]}x{module.kernel_size[0]} kernel"})
        elif isinstance(module, nn.BatchNorm2d):
            layers.append({"name": name, "type": "BatchNorm2D", "detail": f"{module.num_features} channels"})
        elif isinstance(module, nn.MaxPool2d):
            layers.append({"name": name, "type": "MaxPool2D", "detail": f"kernel {module.kernel_size}"})
        elif isinstance(module, nn.Dropout):
            layers.append({"name": name, "type": "Dropout", "detail": f"p={module.p}"})
        elif isinstance(module, nn.Linear):
            layers.append({"name": name, "type": "Linear (Fully Connected)", "detail": f"{module.in_features} -> {module.out_features}"})
    total_params = sum(p.numel() for p in model.parameters())
    return jsonify({"layers": layers, "total_params": total_params, "input_shape": [3, 128, 128], "output_classes": CLASSES})


# ===================== Live federated training (real, over Socket.IO) =====================
class StopFlag:
    def __init__(self):
        self.should_stop = False


active_run_stop_flags = {}


@app.route("/api/training-runs")
@login_required
def training_runs():
    if not current_user.is_admin():
        return jsonify({"error": "Admin only"}), 403
    runs = TrainingRun.query.order_by(TrainingRun.started_at.desc()).limit(20).all()
    return jsonify({
        "runs": [
            {"id": r.id, "hospital_ids": r.hospital_ids, "num_rounds": r.num_rounds,
             "status": r.status, "final_accuracy": r.final_accuracy,
             "started_at": r.started_at.isoformat(),
             "finished_at": r.finished_at.isoformat() if r.finished_at else None}
            for r in runs
        ]
    })


@app.route("/api/live-train", methods=["POST"])
@login_required
def live_train():
    if not current_user.is_admin():
        return jsonify({"error": "Admin only"}), 403

    data = request.get_json(force=True)
    hospital_ids = data.get("hospital_ids", [])
    num_rounds = int(data.get("num_rounds", 5))

    if not hospital_ids or len(hospital_ids) < 2:
        return jsonify({"error": "Select at least 2 hospitals"}), 400

    run = TrainingRun(
        hospital_ids=",".join(map(str, hospital_ids)), num_rounds=num_rounds,
        status="running", triggered_by=current_user.username,
    )
    db.session.add(run)
    db.session.commit()
    run_id = run.id

    stop_flag = StopFlag()
    active_run_stop_flags[run_id] = stop_flag
    last_accuracy = {"value": None}

    def progress_callback(round_num, acc, participants):
        last_accuracy["value"] = acc
        socketio.emit("training_progress", {
            "run_id": run_id, "round": round_num, "accuracy": acc, "participants": participants,
        })

    def background_job():
        from datetime import datetime, timezone
        with app.app_context():
            try:
                live_trainer.run_live_training(
                    hospital_ids=hospital_ids, num_rounds=num_rounds,
                    progress_callback=progress_callback, stop_flag=stop_flag,
                )
                final_run = db.session.get(TrainingRun, run_id)
                final_run.status = "stopped" if stop_flag.should_stop else "completed"
                final_run.final_accuracy = last_accuracy["value"]
                final_run.finished_at = datetime.now(timezone.utc)
                db.session.commit()
                socketio.emit("training_complete", {
                    "run_id": run_id, "status": final_run.status, "final_accuracy": final_run.final_accuracy,
                })
            except Exception as e:
                print(f"Live training failed: {e}")
                failed_run = db.session.get(TrainingRun, run_id)
                failed_run.status = "error"
                failed_run.finished_at = datetime.now(timezone.utc)
                db.session.commit()
                socketio.emit("training_error", {"run_id": run_id, "error": str(e)})
            finally:
                active_run_stop_flags.pop(run_id, None)

    socketio.start_background_task(background_job)
    return jsonify({"ok": True, "run_id": run_id})


@app.route("/api/stop-training", methods=["POST"])
@login_required
def stop_training():
    if not current_user.is_admin():
        return jsonify({"error": "Admin only"}), 403
    data = request.get_json(force=True)
    run_id = data.get("run_id")
    flag = active_run_stop_flags.get(run_id)
    if flag:
        flag.should_stop = True
        return jsonify({"ok": True})
    return jsonify({"error": "No active run with that id"}), 404


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)
