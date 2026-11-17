// Combined UI controller: Image-2 Video Preview + Rings + Chat AND Image-1 2A/2B/3/4 sections

const EMOTION_COLORS = {
  angry: "#dc3545",
  disgust: "#6f42c1",
  fear: "#fd7e14",
  happy: "#198754",
  neutral: "#0d6efd",
  sad: "#6c757d",
  surprise: "#0dcaf0",
};

const DEFAULT_EMOTIONS = {
  angry: 0,
  disgust: 0,
  fear: 0,
  happy: 0,
  neutral: 0,
  sad: 0,
  surprise: 0,
};

const RING_CIRCUMFERENCE = 2 * Math.PI * 34;
let currentPreviewUrl = null;

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setRing(circleId, percent) {
  const circle = document.getElementById(circleId);
  if (!circle) return;
  const clamped = Math.max(0, Math.min(100, Number(percent) || 0));
  circle.style.strokeDasharray = String(RING_CIRCUMFERENCE);
  circle.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - clamped / 100));
}

document.addEventListener("DOMContentLoaded", () => {
  loadQuestions();
  loadHistory();
  loadValidations();
  drawEmotionChart(DEFAULT_EMOTIONS);

  const videoInput = document.getElementById("videoInput");
  if (videoInput) videoInput.addEventListener("change", handleVideoPreview);

  const analyzeForm = document.getElementById("analyzeForm");
  if (analyzeForm) analyzeForm.addEventListener("submit", handleAnalyzeSubmit);

  const validationForm = document.getElementById("validationForm");
  if (validationForm) validationForm.addEventListener("submit", handleValidationSubmit);
});

function handleVideoPreview(e) {
  const file = e.target.files[0];
  const player = document.getElementById("videoPreview");
  const placeholder = document.getElementById("videoPlaceholder");

  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
    currentPreviewUrl = null;
  }

  if (!file) {
    if (player) {
      player.classList.add("d-none");
      player.removeAttribute("src");
    }
    if (placeholder) placeholder.classList.remove("d-none");
    setText("videoBadgeName", "No video selected");
    return;
  }

  currentPreviewUrl = URL.createObjectURL(file);
  if (player) {
    player.src = currentPreviewUrl;
    player.classList.remove("d-none");
  }
  if (placeholder) placeholder.classList.add("d-none");

  const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
  setText("videoBadgeName", `${file.name} (${sizeMb} MB)`);
}

async function loadQuestions() {
  const select = document.getElementById("questionSelect");
  if (!select) return;
  try {
    const res = await fetch("/api/questions");
    const data = await res.json();
    select.innerHTML = '<option value="">-- Select an HR Question --</option>';
    data.questions.forEach((q) => {
      const opt = document.createElement("option");
      opt.value = q.id;
      opt.textContent = `${q.id}. ${q.question}`;
      select.appendChild(opt);
    });
  } catch (err) {
    select.innerHTML = '<option value="">Failed to load questions</option>';
  }
}

async function handleAnalyzeSubmit(e) {
  e.preventDefault();
  const errorBox = document.getElementById("errorBox");
  const loadingBox = document.getElementById("loadingBox");
  const analyzeBtn = document.getElementById("analyzeBtn");

  if (errorBox) errorBox.classList.add("d-none");
  if (loadingBox) loadingBox.classList.remove("d-none");
  if (analyzeBtn) analyzeBtn.disabled = true;

  const formData = new FormData();
  formData.append("candidate_name", document.getElementById("candidateName").value);
  formData.append("question_id", document.getElementById("questionSelect").value);
  formData.append("video", document.getElementById("videoInput").files[0]);

  try {
    const res = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Analysis failed.");
    }
    renderReport(data.report);
    loadHistory();
  } catch (err) {
    if (errorBox) {
      errorBox.textContent = err.message;
      errorBox.classList.remove("d-none");
    }
  } finally {
    if (loadingBox) loadingBox.classList.add("d-none");
    if (analyzeBtn) analyzeBtn.disabled = false;
  }
}

function renderReport(report) {
  const pct = report.emotion_percentages || DEFAULT_EMOTIONS;
  const dominant = report.dominant_emotion || "none";
  const score = report.similarity_score || 0;
  const composure = Math.round((pct.happy || 0) + (pct.neutral || 0));

  // Sidebar & top badges
  setText("kpiEmotion", dominant);
  setText("kpiRelevance", `${score}%`);
  setText("topAttemptBadge", `Attempt: ${report.attempt_id}`);

  // Right Panel Rings & Chat Bubbles
  setRing("emotionRing", composure);
  setText("emotionRingText", `${composure}%`);
  setRing("relevanceRing", score);
  setText("ringRelevanceText", `${score}%`);
  setText("chatCandidateBubble", report.transcript || "(No clear speech transcribed)");
  setText("chatReferenceBubble", report.reference_answer || "");

  // 2A. Facial-Expression Analysis (Image-1 Horizontal Bar Graph + Feedback)
  setText("dominantBadge", `Dominant: ${dominant}`);
  setText("faceStatsText", `Faces detected in ${report.faces_found} of ${report.frames_sampled} sampled frames.`);
  drawEmotionChart(pct);
  renderList("emotionFeedbackList", report.emotion_feedback || []);

  // 2B. Answer Relevance (TF-IDF Cosine)
  setText("similarityValue", `${score}%`);
  setText("relevanceBadge", report.relevance_label || "-");
  const bar = document.getElementById("similarityBar");
  if (bar) {
    bar.style.width = `${Math.min(score, 100)}%`;
    bar.className = "progress-bar " + (score >= 60 ? "bg-success" : score >= 30 ? "bg-warning" : "bg-danger");
  }

  const warnBox = document.getElementById("speechWarningBox");
  if (warnBox) {
    if (report.speech_warning) {
      warnBox.textContent = report.speech_warning;
      warnBox.classList.remove("d-none");
    } else {
      warnBox.classList.add("d-none");
    }
  }

  setText("transcriptText", report.transcript || "(No speech transcribed)");
  setText("referenceText", report.reference_answer || "");
  renderList("answerFeedbackList", report.answer_feedback || []);

  // 3. Human-in-the-Loop Assessor Validation
  const valInput = document.getElementById("valAttemptId");
  if (valInput) valInput.value = report.attempt_id;
  setText("attemptIdBadge", `Attempt: ${report.attempt_id}`);
  setText("validationMsg", "");
}

function drawEmotionChart(percentages) {
  const canvas = document.getElementById("emotionChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const entries = Object.entries( { ...DEFAULT_EMOTIONS, ...percentages });
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);
  ctx.font = "13px system-ui, sans-serif";

  const labelWidth = 82;
  const rightPad = 58;
  const barMaxWidth = w - labelWidth - rightPad;
  const rowHeight = Math.floor((h - 20) / Math.max(entries.length, 1));

  entries.forEach(([emotion, pct], idx) => {
    const y = 14 + idx * rowHeight;
    const barW = (Math.max(0, Math.min(100, pct)) / 100) * barMaxWidth;

    ctx.fillStyle = "#212529";
    ctx.textBaseline = "middle";
    ctx.fillText(emotion, 10, y + 10);

    ctx.fillStyle = "#e9ecef";
    ctx.fillRect(labelWidth, y, barMaxWidth, 18);

    ctx.fillStyle = EMOTION_COLORS[emotion] || "#0d6efd";
    ctx.fillRect(labelWidth, y, barW, 18);

    ctx.fillStyle = "#212529";
    ctx.fillText(`${pct}%`, labelWidth + barMaxWidth + 8, y + 10);
  });
}

function renderList(elementId, items) {
  const ul = document.getElementById(elementId);
  if (!ul) return;
  ul.innerHTML = "";
  items.forEach((txt) => {
    const li = document.createElement("li");
    li.textContent = txt;
    ul.appendChild(li);
  });
}

async function handleValidationSubmit(e) {
  e.preventDefault();
  const attemptId = document.getElementById("valAttemptId").value;
  const assessorName = document.getElementById("assessorName").value;
  const agreementEl = document.querySelector('input[name="agreement"]:checked');
  const notes = document.getElementById("differenceNotes").value;
  const msg = document.getElementById("validationMsg");

  if (!attemptId) {
    if (msg) {
      msg.className = "small text-danger";
      msg.textContent = "Please analyze an interview video first.";
    }
    return;
  }
  if (!agreementEl) return;

  const res = await fetch("/api/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      attempt_id: attemptId,
      assessor_name: assessorName,
      agreement: agreementEl.value,
      difference_notes: notes,
    }),
  });

  const data = await res.json();
  if (res.ok) {
    if (msg) {
      msg.className = "small text-success fw-semibold";
      msg.textContent = `Saved! Agreement rate: ${data.stats.agreement_percent}%`;
    }
    document.getElementById("validationForm").reset();
    document.getElementById("valAttemptId").value = attemptId;
    loadValidations();
  } else if (msg) {
    msg.className = "small text-danger";
    msg.textContent = data.error || "Failed to save validation.";
  }
}

async function loadHistory() {
  const tbody = document.getElementById("historyTableBody");
  if (!tbody) return;
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    const rows = data.history || [];
    setText("kpiAttempts", rows.length);

    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-muted text-center">No attempts yet.</td></tr>';
      return;
    }

    setText("kpiEmotion", rows[0].dominant_emotion || "-");
    setText("kpiRelevance", `${rows[0].similarity_score}%`);

    tbody.innerHTML = "";
    rows.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(r.timestamp)}</td>
        <td>${escapeHtml(r.candidate_name)}</td>
        <td>${escapeHtml(r.question)}</td>
        <td><span class="badge bg-secondary">${escapeHtml(r.dominant_emotion)}</span></td>
        <td>${r.similarity_score}%</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = '<tr><td colspan="5" class="text-danger text-center">Error loading history.</td></tr>';
  }
}

async function loadValidations() {
  const tbody = document.getElementById("validationTableBody");
  if (!tbody) return;
  try {
    const res = await fetch("/api/validations");
    const data = await res.json();
    const stats = data.stats || { agreement_percent: 0, total_reviews: 0 };
    setText("agreementStatsBadge", `Agreement: ${stats.agreement_percent}% (${stats.total_reviews})`);
    setText("kpiAgreement", `${stats.agreement_percent}%`);
    setText("kpiAgreementSub", `${stats.total_reviews} human reviews`);

    const rows = data.validations || [];
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="4" class="text-muted text-center">No validations yet.</td></tr>';
      return;
    }
    tbody.innerHTML = "";
    rows.forEach((v) => {
      const tr = document.createElement("tr");
      const cls = v.agreement === "Yes" ? "bg-success" : "bg-danger";
      tr.innerHTML = `
        <td><code>${escapeHtml(v.attempt_id)}</code></td>
        <td>${escapeHtml(v.assessor_name)}</td>
        <td><span class="badge ${cls}">${escapeHtml(v.agreement)}</span></td>
        <td>${escapeHtml(v.difference_notes)}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-danger text-center">Error loading validations.</td></tr>';
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
