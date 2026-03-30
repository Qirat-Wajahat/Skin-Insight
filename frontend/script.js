/**
 * script.js – Skin Insight Frontend
 * -----------------------------------
 * Handles:
 *  1. User registration modal
 *  2. Camera initialisation and image capture
 *  3. File upload fallback
 *  4. API calls: /predict and /recommend
 *  5. Dynamic rendering of results and product cards
 *  6. Dark mode toggle (persisted in localStorage)
 */

// ── Configuration ──────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:5000";   // Change to your server URL if deployed

// ── DOM References ─────────────────────────────────────────────────────────────
const darkModeToggle     = document.getElementById("darkModeToggle");
const registerModal      = document.getElementById("registerModal");
const registerBtn        = document.getElementById("registerBtn");
const skipRegisterBtn    = document.getElementById("skipRegisterBtn");
const userNameInput      = document.getElementById("userName");
const userEmailInput     = document.getElementById("userEmail");
const registerError      = document.getElementById("registerError");

const startCameraBtn     = document.getElementById("startCameraBtn");
const captureBtn         = document.getElementById("captureBtn");
const retakeBtn          = document.getElementById("retakeBtn");
const videoFeed          = document.getElementById("videoFeed");
const snapshotCanvas     = document.getElementById("snapshotCanvas");
const capturedImage      = document.getElementById("capturedImage");
const videoOverlay       = document.getElementById("videoOverlay");
const cameraStatus       = document.getElementById("cameraStatus");
const fileInput          = document.getElementById("fileInput");
const analyseBtn         = document.getElementById("analyseBtn");
const captureError       = document.getElementById("captureError");

const resultsPlaceholder = document.getElementById("resultsPlaceholder");
const loader             = document.getElementById("loader");
const resultsContent     = document.getElementById("resultsContent");
const predictedClass     = document.getElementById("predictedClass");
const confidenceBar      = document.getElementById("confidenceBar");
const confidenceText     = document.getElementById("confidenceText");
const productList        = document.getElementById("productList");
const totalPrice         = document.getElementById("totalPrice");
const resultsError       = document.getElementById("resultsError");

// ── State ──────────────────────────────────────────────────────────────────────
let mediaStream      = null;   // Active MediaStream from getUserMedia
let capturedBlob     = null;   // Blob of the captured/uploaded image
let userId           = null;   // Registered user id (optional)

// ── 1. Dark Mode ───────────────────────────────────────────────────────────────
function applyDarkMode(enabled) {
  document.body.classList.toggle("dark-mode", enabled);
  darkModeToggle.textContent = enabled ? "☀️" : "🌙";
}

// Load persisted preference
applyDarkMode(localStorage.getItem("darkMode") === "true");

darkModeToggle.addEventListener("click", () => {
  const isDark = document.body.classList.toggle("dark-mode");
  darkModeToggle.textContent = isDark ? "☀️" : "🌙";
  localStorage.setItem("darkMode", isDark);
});

// ── 2. Registration Modal ──────────────────────────────────────────────────────
function closeModal() {
  registerModal.classList.add("hidden");
}

skipRegisterBtn.addEventListener("click", closeModal);

registerBtn.addEventListener("click", async () => {
  registerError.textContent = "";
  const name  = userNameInput.value.trim();
  const email = userEmailInput.value.trim();

  if (!name || !email) {
    registerError.textContent = "Please fill in both fields.";
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email }),
    });
    const data = await res.json();

    if (!res.ok) {
      registerError.textContent = data.error || "Registration failed.";
      return;
    }

    userId = data.user_id;
    closeModal();
  } catch {
    // Backend may not be running – allow guest usage
    closeModal();
  }
});

// ── 3. Camera ──────────────────────────────────────────────────────────────────
startCameraBtn.addEventListener("click", async () => {
  captureError.textContent = "";

  if (mediaStream) {
    // Camera already running – do nothing
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });
    videoFeed.srcObject = mediaStream;
    videoFeed.style.display = "block";
    capturedImage.style.display = "none";
    videoOverlay.classList.add("hidden");
    cameraStatus.textContent = "Camera On";
    cameraStatus.classList.add("active");
    captureBtn.disabled = false;
  } catch (err) {
    captureError.textContent = "Camera access denied or unavailable. Please upload an image instead.";
  }
});

captureBtn.addEventListener("click", () => {
  if (!mediaStream) return;

  const { videoWidth: w, videoHeight: h } = videoFeed;
  snapshotCanvas.width  = w || 640;
  snapshotCanvas.height = h || 480;

  const ctx = snapshotCanvas.getContext("2d");
  ctx.drawImage(videoFeed, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

  // Convert canvas to blob for upload
  snapshotCanvas.toBlob(blob => {
    capturedBlob = blob;
    const url = URL.createObjectURL(blob);
    capturedImage.src = url;
    capturedImage.style.display = "block";
    videoFeed.style.display = "none";

    // Stop camera stream
    stopCamera();

    retakeBtn.style.display = "";
    captureBtn.disabled = true;
    analyseBtn.disabled = false;
  }, "image/jpeg", 0.92);
});

retakeBtn.addEventListener("click", async () => {
  capturedBlob = null;
  capturedImage.style.display = "none";
  retakeBtn.style.display = "none";
  analyseBtn.disabled = true;
  captureError.textContent = "";

  // Restart camera
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });
    videoFeed.srcObject = mediaStream;
    videoFeed.style.display = "block";
    videoOverlay.classList.add("hidden");
    captureBtn.disabled = false;
  } catch {
    captureError.textContent = "Unable to restart camera.";
  }
});

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
    cameraStatus.textContent = "Camera Off";
    cameraStatus.classList.remove("active");
  }
}

// ── 4. File Upload Fallback ────────────────────────────────────────────────────
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;

  capturedBlob = file;
  const url = URL.createObjectURL(file);
  capturedImage.src = url;
  capturedImage.style.display = "block";
  videoFeed.style.display = "none";
  videoOverlay.classList.add("hidden");
  stopCamera();

  retakeBtn.style.display = "";
  captureBtn.disabled = true;
  analyseBtn.disabled = false;
  captureError.textContent = "";
});

// ── 5. Analyse (Predict + Recommend) ──────────────────────────────────────────
analyseBtn.addEventListener("click", async () => {
  if (!capturedBlob) {
    captureError.textContent = "Please capture or upload an image first.";
    return;
  }

  // Show loader, hide other states
  resultsPlaceholder.style.display = "none";
  resultsContent.style.display     = "none";
  loader.style.display              = "flex";
  resultsError.textContent          = "";
  captureError.textContent          = "";
  analyseBtn.disabled               = true;

  try {
    // ── Predict ──────────────────────────────────────────────────────────────
    const formData = new FormData();
    formData.append("file", capturedBlob, "skin_capture.jpg");
    if (userId) formData.append("user_id", userId);

    const predictRes = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    const predictData = await predictRes.json();

    if (!predictRes.ok) {
      throw new Error(predictData.error || "Prediction failed.");
    }

    const { predicted_class: skinProblem, confidence, image_id: imageId } = predictData;

    // ── Recommend ─────────────────────────────────────────────────────────────
    const params = new URLSearchParams({ skin_problem: skinProblem });
    if (imageId) params.append("image_id", imageId);

    const recommendRes = await fetch(`${API_BASE}/recommend?${params}`);
    const recommendData = await recommendRes.json();

    // ── Render Results ────────────────────────────────────────────────────────
    renderResults(skinProblem, confidence, recommendData);

  } catch (err) {
    loader.style.display     = "none";
    resultsPlaceholder.style.display = "";
    resultsError.textContent = err.message || "An unexpected error occurred.";
  } finally {
    analyseBtn.disabled = false;
  }
});

// ── 6. Render Results ──────────────────────────────────────────────────────────
function renderResults(skinProblem, confidence, recommendData) {
  loader.style.display         = "none";
  resultsContent.style.display = "";

  // Predicted class badge
  predictedClass.textContent = skinProblem;

  // Confidence bar animation
  const pct = Math.round(confidence);
  confidenceBar.style.width = `${pct}%`;
  confidenceText.textContent = `${pct}%`;

  // Colour the bar based on confidence level
  if (pct >= 75) {
    confidenceBar.style.background = "var(--clr-success)";
    confidenceText.style.color     = "var(--clr-success)";
  } else if (pct >= 50) {
    confidenceBar.style.background = "var(--clr-accent)";
    confidenceText.style.color     = "var(--clr-accent)";
  } else {
    confidenceBar.style.background = "var(--clr-error)";
    confidenceText.style.color     = "var(--clr-error)";
  }

  // Product cards
  productList.innerHTML = "";

  if (recommendData.products && recommendData.products.length > 0) {
    recommendData.products.forEach(product => {
      const card = document.createElement("div");
      card.className = "product-card";
      card.innerHTML = `
        <div class="product-name">${escapeHtml(product.product_name)}</div>
        <div class="product-brand">${escapeHtml(product.brand)}</div>
        <div class="product-price">Rs ${product.price.toLocaleString()}</div>
      `;
      productList.appendChild(card);
    });

    totalPrice.textContent = `Rs ${recommendData.total_price.toLocaleString()}`;
  } else {
    productList.innerHTML = `<p style="color:var(--clr-text-muted);font-size:.88rem;">No products found for this condition.</p>`;
    totalPrice.textContent = "Rs 0";
  }
}

// ── Utility: Escape HTML to prevent XSS ────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
