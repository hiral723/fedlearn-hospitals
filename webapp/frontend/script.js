const CLASS_COLORS = {
  COVID: "var(--covid)", Normal: "var(--normal)",
  Lung_Opacity: "var(--opacity)", Viral_Pneumonia: "var(--pneumonia)",
};
const CLUSTER_COLORS = ["#4ca583", "#bb8a4a", "#8a7ab5", "#b56a5f", "#5f9ab0", "#7fb896"];

let currentUser = null;

// ===================== Boot: auth + role-based nav =====================
async function boot() {
  const res = await fetch("/api/me");
  if (res.status === 401) { window.location.href = "/login"; return; }
  currentUser = await res.json();

  document.getElementById("topBarUser").textContent = currentUser.display_name || currentUser.username;
  document.getElementById("topBarRole").textContent = currentUser.role;

  const role = currentUser.role;
  document.querySelectorAll(".admin-only").forEach((el) => { el.style.display = role === "admin" ? "" : "none"; });
  document.querySelectorAll(".hospital-only").forEach((el) => { el.style.display = role === "hospital" ? "" : "none"; });
  document.querySelectorAll(".patient-only").forEach((el) => { el.style.display = role === "patient" ? "" : "none"; });
  document.querySelectorAll(".hospital-admin-only").forEach((el) => {
    el.style.display = (role === "hospital" || role === "admin") ? "" : "none";
  });

  loadOverviewStat();
}
boot();

document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.page).classList.add("active");

    if (btn.dataset.page === "history" && !historyLoaded) loadHistory();
    if (btn.dataset.page === "livetrain" && !liveTrainLoaded) loadLiveTrainPage();
    if (btn.dataset.page === "allhospitals" && !allHospitalsLoaded) loadAllHospitals();
    if (btn.dataset.page === "architecture" && !architectureLoaded) loadArchitecture();
    if (btn.dataset.page === "results" && !resultsLoaded) loadResults();
    if (btn.dataset.page === "patientrecords") loadPatientRecords();
  });
});

async function loadOverviewStat() {
  try {
    const res = await fetch("/api/results");
    const data = await res.json();
    let best = null;
    if (data.clusters) {
      Object.values(data.clusters).forEach((h) => {
        const finalAcc = h.accuracy[h.accuracy.length - 1];
        if (best === null || finalAcc > best) best = finalAcc;
      });
    }
    if (best !== null) document.getElementById("statBestAcc").textContent = (best * 100).toFixed(1) + "%";
  } catch (err) { /* non-critical */ }
}

function probBarsHtml(probabilities) {
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  let html = "";
  for (const [className, prob] of sorted) {
    html += `<div class="prob-row">
      <span>${className.replace("_", " ")}</span>
      <div class="prob-track"><div class="prob-fill" style="width:${(prob * 100).toFixed(1)}%; background:${CLASS_COLORS[className] || "#888"}"></div></div>
      <span>${(prob * 100).toFixed(1)}%</span></div>`;
  }
  return html;
}

function heatmapGridHtml(heatmaps, probabilities, predictedClass) {
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  let html = "";
  sorted.forEach(([className, prob]) => {
    html += `<div class="heatmap-cell${className === predictedClass ? " predicted" : ""}">
      <img src="${heatmaps[className]}" alt="Grad-CAM for ${className}" />
      <div class="heatmap-cell-label">${className.replace("_", " ")}${className === predictedClass ? " (predicted)" : ""}</div>
      <div class="heatmap-cell-prob">${(prob * 100).toFixed(1)}% confidence</div></div>`;
  });
  return html;
}

// ===================== Classify page (hospital/admin) =====================
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const dropzoneText = document.getElementById("dropzoneText");
const predictBtn = document.getElementById("predictBtn");
const resultCard = document.getElementById("resultCard");
const predictError = document.getElementById("predictError");
let selectedFile = null;

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault(); dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => { preview.src = e.target.result; preview.hidden = false; dropzoneText.textContent = file.name; };
  reader.readAsDataURL(file);
  predictBtn.disabled = false;
  resultCard.hidden = true;
  predictError.hidden = true;
}

predictBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  predictBtn.disabled = true;
  predictBtn.textContent = "Classifying...";
  predictError.hidden = true;
  resultCard.hidden = true;

  const formData = new FormData();
  formData.append("image", selectedFile);
  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed");

    document.getElementById("predictedClass").textContent = data.predicted_class.replace("_", " ");
    document.getElementById("confidenceValue").textContent = (data.confidence * 100).toFixed(1) + "%";
    document.getElementById("probBars").innerHTML = probBarsHtml(data.probabilities);
    document.getElementById("heatmapGrid").innerHTML = heatmapGridHtml(data.heatmaps, data.probabilities, data.predicted_class);
    resultCard.hidden = false;
  } catch (err) {
    predictError.textContent = err.message;
    predictError.hidden = false;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Classify Image";
  }
});

// ===================== My History (hospital) =====================
let historyLoaded = false;
async function loadHistory() {
  const errorBox = document.getElementById("historyError");
  errorBox.hidden = true;
  try {
    const res = await fetch("/api/my-history");
    const data = await res.json();
    if (data.error) { errorBox.textContent = data.error; errorBox.hidden = false; return; }

    const wrap = document.getElementById("historyTableWrap");
    if (data.history.length === 0) {
      wrap.innerHTML = '<p class="hint">No predictions yet.</p>';
    } else {
      let html = '<table class="history-table"><tr><th>Predicted Class</th><th>Confidence</th><th>Source</th><th>Timestamp</th></tr>';
      data.history.forEach((row) => {
        const dt = new Date(row.timestamp).toLocaleString();
        html += `<tr><td>${row.predicted_class.replace("_", " ")}</td><td>${(row.confidence * 100).toFixed(1)}%</td><td>${row.source_filename || "—"}</td><td>${dt}</td></tr>`;
      });
      html += "</table>";
      wrap.innerHTML = html;
    }
    historyLoaded = true;
  } catch (err) {
    errorBox.textContent = "Could not load history: " + err.message;
    errorBox.hidden = false;
  }
}

// ===================== Batch upload (hospital/admin) =====================
const batchDropzone = document.getElementById("batchDropzone");
const batchFileInput = document.getElementById("batchFileInput");
const batchDropzoneText = document.getElementById("batchDropzoneText");
const batchSubmitBtn = document.getElementById("batchSubmitBtn");
let selectedBatchFile = null;

batchDropzone.addEventListener("click", () => batchFileInput.click());
batchDropzone.addEventListener("dragover", (e) => { e.preventDefault(); batchDropzone.classList.add("dragover"); });
batchDropzone.addEventListener("dragleave", () => batchDropzone.classList.remove("dragover"));
batchDropzone.addEventListener("drop", (e) => {
  e.preventDefault(); batchDropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleBatchFile(e.dataTransfer.files[0]);
});
batchFileInput.addEventListener("change", () => { if (batchFileInput.files.length) handleBatchFile(batchFileInput.files[0]); });

function handleBatchFile(file) {
  selectedBatchFile = file;
  batchDropzoneText.textContent = file.name;
  batchSubmitBtn.disabled = false;
  document.getElementById("batchResultCard").hidden = true;
  document.getElementById("batchError").hidden = true;
}

batchSubmitBtn.addEventListener("click", async () => {
  if (!selectedBatchFile) return;
  const errorBox = document.getElementById("batchError");
  errorBox.hidden = true;
  batchSubmitBtn.disabled = true;
  batchSubmitBtn.textContent = "Classifying dataset...";

  const formData = new FormData();
  formData.append("dataset", selectedBatchFile);
  try {
    const res = await fetch("/api/batch-classify", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Batch classification failed");

    const summary = document.getElementById("batchSummary");
    summary.innerHTML = "";
    const totalCard = document.createElement("div");
    totalCard.className = "summary-card";
    totalCard.innerHTML = `<div class="value">${data.total}</div><div class="label">Images Classified</div>`;
    summary.appendChild(totalCard);
    Object.entries(data.class_counts).forEach(([className, count]) => {
      const card = document.createElement("div");
      card.className = "summary-card";
      card.innerHTML = `<div class="value">${count}</div><div class="label">${className.replace("_", " ")}</div>`;
      summary.appendChild(card);
    });

    let html = '<table class="batch-table"><tr><th>Filename</th><th>Predicted Class</th><th>Confidence</th></tr>';
    data.results.forEach((r) => {
      if (r.error) {
        html += `<tr><td>${r.filename}</td><td colspan="2" style="color:var(--red)">Error: ${r.error}</td></tr>`;
      } else {
        html += `<tr><td>${r.filename}</td><td>${r.predicted_class.replace("_", " ")}</td><td>${(r.confidence * 100).toFixed(1)}%</td></tr>`;
      }
    });
    html += "</table>";
    document.getElementById("batchTableWrap").innerHTML = html;
    document.getElementById("batchResultCard").hidden = false;
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  } finally {
    batchSubmitBtn.disabled = false;
    batchSubmitBtn.textContent = "Classify Dataset";
  }
});

// ===================== Patient: upload X-ray =====================
const patXrayDropzone = document.getElementById("patXrayDropzone");
const patXrayInput = document.getElementById("patXrayInput");
const patXrayPreview = document.getElementById("patXrayPreview");
const patXrayBtn = document.getElementById("patXrayBtn");
let selectedPatXrayFile = null;
let currentPatientRecordId = null;

patXrayDropzone.addEventListener("click", () => patXrayInput.click());
patXrayDropzone.addEventListener("dragover", (e) => { e.preventDefault(); patXrayDropzone.classList.add("dragover"); });
patXrayDropzone.addEventListener("dragleave", () => patXrayDropzone.classList.remove("dragover"));
patXrayDropzone.addEventListener("drop", (e) => {
  e.preventDefault(); patXrayDropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handlePatXrayFile(e.dataTransfer.files[0]);
});
patXrayInput.addEventListener("change", () => { if (patXrayInput.files.length) handlePatXrayFile(patXrayInput.files[0]); });

function handlePatXrayFile(file) {
  selectedPatXrayFile = file;
  const reader = new FileReader();
  reader.onload = (e) => { patXrayPreview.src = e.target.result; patXrayPreview.hidden = false; };
  reader.readAsDataURL(file);
  patXrayBtn.disabled = false;
}

patXrayBtn.addEventListener("click", async () => {
  if (!selectedPatXrayFile) return;
  const errorBox = document.getElementById("patXrayError");
  errorBox.hidden = true;
  patXrayBtn.disabled = true;
  patXrayBtn.textContent = "Classifying...";

  const formData = new FormData();
  formData.append("image", selectedPatXrayFile);
  try {
    const res = await fetch("/api/patient/upload-xray", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");

    currentPatientRecordId = data.record_id;
    document.getElementById("patPredictedClass").textContent = data.predicted_class.replace("_", " ");
    document.getElementById("patConfidenceValue").textContent = (data.confidence * 100).toFixed(1) + "%";
    document.getElementById("patProbBars").innerHTML = probBarsHtml(data.probabilities);
    document.getElementById("patXrayResultCard").hidden = false;
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  } finally {
    patXrayBtn.disabled = false;
    patXrayBtn.textContent = "Classify My X-Ray";
  }
});

// ===================== Patient: upload report (OCR) =====================
const patReportDropzone = document.getElementById("patReportDropzone");
const patReportInput = document.getElementById("patReportInput");
const patReportDropzoneText = document.getElementById("patReportDropzoneText");
const patReportBtn = document.getElementById("patReportBtn");
let selectedPatReportFile = null;

patReportDropzone.addEventListener("click", () => patReportInput.click());
patReportDropzone.addEventListener("dragover", (e) => { e.preventDefault(); patReportDropzone.classList.add("dragover"); });
patReportDropzone.addEventListener("dragleave", () => patReportDropzone.classList.remove("dragover"));
patReportDropzone.addEventListener("drop", (e) => {
  e.preventDefault(); patReportDropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handlePatReportFile(e.dataTransfer.files[0]);
});
patReportInput.addEventListener("change", () => { if (patReportInput.files.length) handlePatReportFile(patReportInput.files[0]); });

function handlePatReportFile(file) {
  selectedPatReportFile = file;
  patReportDropzoneText.textContent = file.name;
  patReportBtn.disabled = false;
}

patReportBtn.addEventListener("click", async () => {
  if (!selectedPatReportFile) return;
  const errorBox = document.getElementById("patReportError");
  errorBox.hidden = true;
  patReportBtn.disabled = true;
  patReportBtn.textContent = "Extracting text (this can take a moment)...";

  const formData = new FormData();
  formData.append("report", selectedPatReportFile);
  if (currentPatientRecordId) formData.append("record_id", currentPatientRecordId);

  try {
    const res = await fetch("/api/patient/upload-report", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Report upload failed");

    const keywordsEl = document.getElementById("patReportKeywords");
    keywordsEl.innerHTML = data.keywords_found.length
      ? data.keywords_found.map((k) => `<span class="keyword-chip">${k.replace("_", " ")}</span>`).join("")
      : '<span class="hint small">No known disease-class keywords detected in the text.</span>';
    document.getElementById("patReportText").textContent = data.report_text || "(no text detected)";
    document.getElementById("patReportResultCard").hidden = false;
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  } finally {
    patReportBtn.disabled = false;
    patReportBtn.textContent = "Extract Report Text";
  }
});

// ===================== Patient: my records =====================
async function loadPatientRecords() {
  const errorBox = document.getElementById("patientRecordsError");
  errorBox.hidden = true;
  document.getElementById("patientRecordDetailCard").hidden = true;
  try {
    const res = await fetch("/api/patient/my-records");
    const data = await res.json();
    if (data.error) { errorBox.textContent = data.error; errorBox.hidden = false; return; }

    const wrap = document.getElementById("patientRecordsWrap");
    if (data.records.length === 0) {
      wrap.innerHTML = '<p class="hint">No records yet. Head to "Upload X-Ray & Report" to get started.</p>';
      return;
    }
    wrap.innerHTML = "";
    data.records.forEach((r) => {
      const div = document.createElement("div");
      div.className = "record-card";
      const dt = new Date(r.created_at).toLocaleString();
      div.innerHTML = `
        <div class="record-card-title">${r.predicted_class ? r.predicted_class.replace("_", " ") : "Report only"}${r.confidence ? " (" + (r.confidence * 100).toFixed(1) + "%)" : ""}</div>
        <div class="record-card-meta">${r.report_filename ? "Report attached" : "No report"} &middot; ${dt}</div>`;
      div.addEventListener("click", () => loadPatientRecordDetail(r.id));
      wrap.appendChild(div);
    });
  } catch (err) {
    errorBox.textContent = "Could not load records: " + err.message;
    errorBox.hidden = false;
  }
}

async function loadPatientRecordDetail(recordId) {
  const res = await fetch(`/api/patient/record/${recordId}`);
  const data = await res.json();
  if (data.error) return;

  const card = document.getElementById("patientRecordDetailCard");
  const detail = document.getElementById("patientRecordDetail");
  let html = "";

  if (data.predicted_class) {
    html += `<h3 class="section-label">X-Ray Prediction: ${data.predicted_class.replace("_", " ")} (${(data.confidence * 100).toFixed(1)}%)</h3>`;
    if (data.heatmaps && data.probabilities) {
      html += `<div class="heatmap-grid">${heatmapGridHtml(data.heatmaps, data.probabilities, data.predicted_class)}</div>`;
    }
  }
  if (data.report_text) {
    html += `<h3 class="section-label" style="margin-top:1.5rem;">Report Text</h3>`;
    html += data.report_keywords.length
      ? `<div class="network-legend" style="margin-bottom:0.5rem;">${data.report_keywords.map((k) => `<span class="keyword-chip">${k.replace("_", " ")}</span>`).join("")}</div>`
      : "";
    html += `<p class="hint" style="white-space:pre-wrap;">${data.report_text}</p>`;
  }

  detail.innerHTML = html;
  card.hidden = false;
  card.scrollIntoView({ behavior: "smooth" });
}

// ===================== Live Training (admin) =====================
let liveTrainLoaded = false;
let selectedHospitalIds = new Set();
let liveChart = null;
let liveSocket = null;
let currentRunId = null;

async function loadLiveTrainPage() {
  try {
    const res = await fetch("/api/hospitals");
    const data = await res.json();
    const picker = document.getElementById("hospitalPicker");
    picker.innerHTML = "";
    data.hospitals.forEach((h) => {
      const chip = document.createElement("label");
      chip.className = "hospital-chip";
      chip.innerHTML = `<input type="checkbox" value="${h.id}" /> H${h.id}`;
      const checkbox = chip.querySelector("input");
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) { selectedHospitalIds.add(h.id); chip.classList.add("selected"); }
        else { selectedHospitalIds.delete(h.id); chip.classList.remove("selected"); }
      });
      picker.appendChild(chip);
    });

    initLiveChart();
    connectLiveSocket();
    loadTrainingRuns();

    document.getElementById("startTrainBtn").addEventListener("click", startLiveTraining);
    document.getElementById("stopTrainBtn").addEventListener("click", stopLiveTraining);
    liveTrainLoaded = true;
  } catch (err) {
    document.getElementById("trainError").textContent = "Could not load: " + err.message;
    document.getElementById("trainError").hidden = false;
  }
}

function initLiveChart() {
  if (liveChart) liveChart.destroy();
  liveChart = new Chart(document.getElementById("liveChart"), {
    type: "line",
    data: { datasets: [{ label: "Global Accuracy", data: [], borderColor: "#4ca583", backgroundColor: "#4ca583", tension: 0.15 }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      scales: { x: { type: "linear", title: { display: true, text: "Round" } }, y: { min: 0, max: 1, title: { display: true, text: "Accuracy" } } },
      plugins: { legend: { display: false } },
    },
  });
}

function connectLiveSocket() {
  liveSocket = io();
  liveSocket.on("training_progress", (data) => {
    if (data.run_id !== currentRunId) return;
    document.getElementById("liveRoundLabel").textContent = data.round;
    document.getElementById("liveAccLabel").textContent = (data.accuracy * 100).toFixed(2) + "%";
    liveChart.data.datasets[0].data.push({ x: data.round, y: data.accuracy });
    liveChart.update("none");
    const participantsStr = data.participants.length > 0 ? data.participants.join(", ") : "none (initial)";
    logLiveEvent(`Round ${data.round}: hospitals [${participantsStr}] trained locally, weights averaged → accuracy ${(data.accuracy * 100).toFixed(1)}%`);
  });
  liveSocket.on("training_complete", (data) => {
    if (data.run_id !== currentRunId) return;
    document.getElementById("liveStatusLabel").textContent = data.status;
    logLiveEvent(`Training ${data.status}. Final accuracy: ${data.final_accuracy !== null ? (data.final_accuracy * 100).toFixed(2) + "%" : "n/a"}`);
    document.getElementById("startTrainBtn").disabled = false;
    document.getElementById("stopTrainBtn").disabled = true;
    loadTrainingRuns();
  });
  liveSocket.on("training_error", (data) => {
    if (data.run_id !== currentRunId) return;
    logLiveEvent("ERROR: " + data.error);
    document.getElementById("startTrainBtn").disabled = false;
    document.getElementById("stopTrainBtn").disabled = true;
  });
}

function logLiveEvent(message) {
  const log = document.getElementById("liveEventLog");
  const entry = document.createElement("div");
  entry.className = "event-log-entry";
  entry.textContent = message;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

async function startLiveTraining() {
  const errorBox = document.getElementById("trainError");
  errorBox.hidden = true;
  if (selectedHospitalIds.size < 2) {
    errorBox.textContent = "Select at least 2 hospitals.";
    errorBox.hidden = false;
    return;
  }
  const numRounds = parseInt(document.getElementById("numRoundsSelect").value, 10);
  initLiveChart();
  document.getElementById("liveEventLog").innerHTML = "";
  document.getElementById("liveRoundLabel").textContent = "0";
  document.getElementById("liveAccLabel").textContent = "—";
  document.getElementById("liveStatusLabel").textContent = "Starting…";

  try {
    const res = await fetch("/api/live-train", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hospital_ids: Array.from(selectedHospitalIds), num_rounds: numRounds }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Could not start training");
    currentRunId = data.run_id;
    document.getElementById("liveStatusLabel").textContent = "Running";
    document.getElementById("startTrainBtn").disabled = true;
    document.getElementById("stopTrainBtn").disabled = false;
    logLiveEvent(`Started training run #${data.run_id} with hospitals [${Array.from(selectedHospitalIds).join(", ")}], ${numRounds} rounds.`);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  }
}

async function stopLiveTraining() {
  if (currentRunId === null) return;
  await fetch("/api/stop-training", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: currentRunId }),
  });
  logLiveEvent("Stop requested…");
}

async function loadTrainingRuns() {
  try {
    const res = await fetch("/api/training-runs");
    const data = await res.json();
    const wrap = document.getElementById("runsTableWrap");
    if (!data.runs || data.runs.length === 0) { wrap.innerHTML = '<p class="hint">No training runs yet.</p>'; return; }
    let html = '<table class="runs-table"><tr><th>ID</th><th>Hospitals</th><th>Rounds</th><th>Status</th><th>Final Accuracy</th><th>Started</th></tr>';
    data.runs.forEach((r) => {
      const acc = r.final_accuracy !== null ? (r.final_accuracy * 100).toFixed(1) + "%" : "—";
      html += `<tr><td>${r.id}</td><td>${r.hospital_ids}</td><td>${r.num_rounds}</td>
        <td><span class="status-pill ${r.status}">${r.status}</span></td><td>${acc}</td>
        <td>${new Date(r.started_at).toLocaleString()}</td></tr>`;
    });
    html += "</table>";
    wrap.innerHTML = html;
  } catch (err) { /* non-critical */ }
}

// ===================== All Hospitals (admin) =====================
let allHospitalsLoaded = false;
async function loadAllHospitals() {
  const errorBox = document.getElementById("allHospitalsError");
  errorBox.hidden = true;
  try {
    const res = await fetch("/api/hospitals");
    const data = await res.json();
    let html = '<table class="history-table"><tr><th>Hospital</th><th>Cluster</th><th>Training Images</th><th>Predictions Logged</th></tr>';
    data.hospitals.forEach((h) => {
      html += `<tr><td>${h.name}</td><td>${h.cluster !== null ? "Cluster " + h.cluster : "—"}</td><td>${h.total_images}</td><td>${h.prediction_count}</td></tr>`;
    });
    html += "</table>";
    document.getElementById("allHospitalsTableWrap").innerHTML = html;
    allHospitalsLoaded = true;
  } catch (err) {
    errorBox.textContent = "Could not load hospitals: " + err.message;
    errorBox.hidden = false;
  }
}

// ===================== Architecture =====================
let architectureLoaded = false;
async function loadArchitecture() {
  const errorBox = document.getElementById("archError");
  errorBox.hidden = true;
  try {
    const res = await fetch("/api/architecture");
    const data = await res.json();
    if (data.error) { errorBox.textContent = data.error; errorBox.hidden = false; return; }

    const summary = document.getElementById("archSummary");
    summary.innerHTML = "";
    [[data.total_params.toLocaleString(), "Total Parameters"], [data.input_shape.join(" \u00D7 "), "Input Shape"],
     [data.output_classes.length, "Output Classes"], [data.layers.length, "Layers"]].forEach(([value, label]) => {
      const div = document.createElement("div");
      div.className = "stat-card";
      div.innerHTML = `<div class="stat-value">${value}</div><div class="stat-label">${label}</div>`;
      summary.appendChild(div);
    });

    const diagram = document.getElementById("archDiagram");
    diagram.innerHTML = "";
    data.layers.forEach((layer, idx) => {
      const row = document.createElement("div");
      row.className = "arch-layer " + layer.type.toLowerCase().replace(/[^a-z]/g, "");
      row.innerHTML = `<span class="arch-layer-index">${idx + 1}</span><span class="arch-layer-type">${layer.type}</span><span class="arch-layer-detail">${layer.detail}</span>`;
      diagram.appendChild(row);
    });
    architectureLoaded = true;
  } catch (err) {
    errorBox.textContent = "Could not load architecture: " + err.message;
    errorBox.hidden = false;
  }
}

// ===================== Results =====================
let resultsLoaded = false;
async function loadResults() {
  const errorBox = document.getElementById("resultsError");
  errorBox.hidden = true;
  try {
    const res = await fetch("/api/results");
    const data = await res.json();
    if (data.baseline || data.clusters) { renderAccuracyChart(data); renderSummaryCards(data); }

    const metricsRes = await fetch("/api/metrics");
    const metricsData = await metricsRes.json();
    if (!metricsData.error) { renderConfusionMatrix(metricsData); renderMetricsChart(metricsData); }
    resultsLoaded = true;
  } catch (err) {
    errorBox.textContent = "Could not load results: " + err.message;
    errorBox.hidden = false;
  }
}

function renderAccuracyChart(data) {
  const datasets = [];
  if (data.baseline) {
    datasets.push({ label: "Baseline (1 global model)", data: data.baseline.round.map((r, i) => ({ x: r, y: data.baseline.accuracy[i] })), borderColor: "#8fa39b", borderDash: [6, 4], backgroundColor: "#8fa39b", tension: 0.15 });
  }
  if (data.clusters) {
    Object.entries(data.clusters).forEach(([cid, h], idx) => {
      datasets.push({ label: `Cluster ${cid}`, data: h.round.map((r, i) => ({ x: r, y: h.accuracy[i] })), borderColor: CLUSTER_COLORS[idx % CLUSTER_COLORS.length], backgroundColor: CLUSTER_COLORS[idx % CLUSTER_COLORS.length], tension: 0.15 });
    });
  }
  new Chart(document.getElementById("resultsChart"), {
    type: "line", data: { datasets },
    options: { responsive: true, maintainAspectRatio: false,
      scales: { x: { type: "linear", title: { display: true, text: "Federated Round" } }, y: { title: { display: true, text: "Centralized Test Accuracy" }, min: 0, max: 1 } },
      plugins: { legend: { position: "bottom" } } },
  });
}

function renderSummaryCards(data) {
  const container = document.getElementById("summaryCards");
  container.innerHTML = "";
  if (data.baseline) {
    const finalAcc = data.baseline.accuracy[data.baseline.accuracy.length - 1];
    const div = document.createElement("div");
    div.className = "summary-card";
    div.innerHTML = `<div class="value">${(finalAcc * 100).toFixed(1)}%</div><div class="label">Baseline Final Accuracy</div>`;
    container.appendChild(div);
  }
  if (data.clusters) {
    Object.entries(data.clusters).forEach(([cid, h]) => {
      const finalAcc = h.accuracy[h.accuracy.length - 1];
      const div = document.createElement("div");
      div.className = "summary-card";
      div.innerHTML = `<div class="value">${(finalAcc * 100).toFixed(1)}%</div><div class="label">Cluster ${cid} Final Accuracy</div>`;
      container.appendChild(div);
    });
  }
}

function renderConfusionMatrix(metrics) {
  const wrap = document.getElementById("confusionMatrixWrap");
  const classes = metrics.classes;
  const matrix = metrics.confusion_matrix;
  const maxVal = Math.max(...matrix.flat());
  let html = '<table class="confusion-table"><tr><th>True \\ Pred</th>';
  classes.forEach((c) => (html += `<th>${c.replace("_", " ")}</th>`));
  html += "</tr>";
  matrix.forEach((row, i) => {
    html += `<tr><th>${classes[i].replace("_", " ")}</th>`;
    row.forEach((val, j) => {
      const intensity = maxVal > 0 ? val / maxVal : 0;
      const bg = i === j ? `rgba(76, 165, 131, ${0.15 + intensity * 0.65})` : `rgba(181, 106, 95, ${intensity * 0.35})`;
      html += `<td class="${i === j ? "diag" : ""}" style="background:${bg}">${val}</td>`;
    });
    html += "</tr>";
  });
  html += "</table>";
  wrap.innerHTML = html;
}

function renderMetricsChart(metrics) {
  const classes = metrics.classes;
  new Chart(document.getElementById("metricsChart"), {
    type: "bar",
    data: { labels: classes.map((c) => c.replace("_", " ")),
      datasets: [
        { label: "Precision", data: classes.map((c) => metrics.per_class[c].precision), backgroundColor: "#4ca583" },
        { label: "Recall", data: classes.map((c) => metrics.per_class[c].recall), backgroundColor: "#bb8a4a" },
        { label: "F1 Score", data: classes.map((c) => metrics.per_class[c].f1), backgroundColor: "#8a7ab5" },
      ] },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1 } }, plugins: { legend: { position: "bottom" } } },
  });
}
