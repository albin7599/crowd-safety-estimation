// camera.js - Camera Fine-Tuning and Active Model Management

document.addEventListener("DOMContentLoaded", () => {
  const finetuneForm = document.getElementById("finetuneForm");
  const clipFile = document.getElementById("clipFile");
  const secondsInput = document.getElementById("secondsInput");
  const epochsInput = document.getElementById("epochsInput");
  const ftSubmitBtn = document.getElementById("ftSubmitBtn");
  const ftProgressBox = document.getElementById("ftProgressBox");
  const ftStatusMessage = document.getElementById("ftStatusMessage");
  const ftPercentText = document.getElementById("ftPercentText");
  const ftProgressBar = document.getElementById("ftProgressBar");

  const activeModelBadge = document.getElementById("activeModelBadge");
  const loadedModelName = document.getElementById("loadedModelName");
  const loadedModelThresholds = document.getElementById("loadedModelThresholds");
  const selectDefaultBtn = document.getElementById("selectDefaultBtn");
  const selectCameraBtn = document.getElementById("selectCameraBtn");
  const camModelAvailabilityText = document.getElementById("camModelAvailabilityText");

  let statusPollTimer = null;

  async function checkCameraStatus() {
    try {
      const res = await fetch("/camera/status");
      if (!res.ok) return;
      const data = await res.json();

      // Update Active Model HUD
      const isCam = data.active_model === "camera";
      activeModelBadge.textContent = isCam ? "Camera Model" : "Default UCSD";
      loadedModelName.textContent = isCam ? "Camera-Specific Model (predictor_cam.pt)" : "Default Baseline (UCSD Ped2)";

      const cfg = data.config || {};
      const pix = cfg.pix_thr ? Number(cfg.pix_thr).toFixed(5) : "N/A";
      const frame = cfg.frame_thr ? Number(cfg.frame_thr).toFixed(5) : "N/A";
      loadedModelThresholds.textContent = `Pixel Thr: ${pix} | Frame Thr: ${frame}`;

      // Update Switch Buttons
      if (data.camera_model_available) {
        selectCameraBtn.disabled = false;
        camModelAvailabilityText.textContent = "Camera model is available on disk.";
        camModelAvailabilityText.style.color = "var(--ok-green)";
      } else {
        selectCameraBtn.disabled = true;
        camModelAvailabilityText.textContent = "Camera model not yet trained. Upload a normal clip to create it.";
        camModelAvailabilityText.style.color = "var(--text-dim)";
      }

      if (isCam) {
        selectCameraBtn.className = "btn btn-primary";
        selectDefaultBtn.className = "btn btn-secondary";
      } else {
        selectDefaultBtn.className = "btn btn-primary";
        selectCameraBtn.className = "btn btn-secondary";
      }

      // Handle Fine-tune in progress
      if (data.status === "running") {
        ftProgressBox.style.display = "block";
        ftStatusMessage.textContent = data.message || "Fine-tuning camera predictor...";
        ftPercentText.textContent = `${data.progress || 0}%`;
        ftProgressBar.style.width = `${data.progress || 0}%`;
        ftSubmitBtn.disabled = true;
      } else if (data.status === "done" && ftProgressBox.style.display === "block") {
        ftProgressBox.style.display = "block";
        ftStatusMessage.textContent = data.message || "Camera fine-tuning completed!";
        ftPercentText.textContent = "100%";
        ftProgressBar.style.width = "100%";
        ftSubmitBtn.disabled = false;
      } else if (data.status === "error" && ftProgressBox.style.display === "block") {
        ftProgressBox.style.display = "block";
        ftStatusMessage.textContent = data.message || "Fine-tuning encountered an error.";
        ftProgressBar.style.backgroundColor = "var(--alert-red)";
        ftSubmitBtn.disabled = false;
      }

    } catch (err) {
      console.error("Camera status fetch failed:", err);
    }
  }

  // Switch Model Button Handlers
  selectDefaultBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/camera/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "default" })
      });
      const data = await res.json();
      if (res.ok) {
        checkCameraStatus();
      } else {
        alert(data.error || "Failed to switch model");
      }
    } catch (e) {
      alert("Error switching model");
    }
  });

  selectCameraBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/camera/switch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "camera" })
      });
      const data = await res.json();
      if (res.ok) {
        checkCameraStatus();
      } else {
        alert(data.error || "Failed to switch model");
      }
    } catch (e) {
      alert("Error switching model");
    }
  });

  // Handle Fine-tuning Form Submit
  finetuneForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!clipFile.files || clipFile.files.length === 0) {
      alert("Please select a video clip containing only normal activity.");
      return;
    }

    const formData = new FormData();
    formData.append("clip", clipFile.files[0]);
    formData.append("seconds", secondsInput.value || "40");
    formData.append("epochs", epochsInput.value || "15");

    ftSubmitBtn.disabled = true;
    ftProgressBox.style.display = "block";
    ftStatusMessage.textContent = "Uploading video clip...";
    ftProgressBar.style.width = "10%";
    ftPercentText.textContent = "10%";

    try {
      const res = await fetch("/camera/finetune", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (!res.ok) {
        alert(data.error || "Failed to start fine-tuning");
        ftSubmitBtn.disabled = false;
        ftProgressBox.style.display = "none";
      } else {
        ftStatusMessage.textContent = data.message;
      }
    } catch (err) {
      alert("Network error starting fine-tuning");
      ftSubmitBtn.disabled = false;
      ftProgressBox.style.display = "none";
    }
  });

  checkCameraStatus();
  statusPollTimer = setInterval(checkCameraStatus, 2500);
});
