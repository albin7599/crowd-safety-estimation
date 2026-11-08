// upload.js - Interactive Drag-and-Drop and Form Validation for IADL-CDE

document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const videoInput = document.getElementById("videoInput");
  const filePreview = document.getElementById("filePreview");
  const previewFilename = document.getElementById("previewFilename");
  const previewFilesize = document.getElementById("previewFilesize");
  const removeFileBtn = document.getElementById("removeFileBtn");
  const uploadForm = document.getElementById("uploadForm");
  const submitBtn = document.getElementById("submitBtn");
  const calibNormalCheckbox = document.getElementById("calibNormalCheckbox");
  const calibSecondsInput = document.getElementById("calibSecondsInput");
  const calibSecondsGroup = document.getElementById("calibSecondsGroup");
  const uploadProgressContainer = document.getElementById("uploadProgressContainer");
  const uploadProgressBar = document.getElementById("uploadProgressBar");
  const uploadPercentText = document.getElementById("uploadPercentText");
  const uploadStatusText = document.getElementById("uploadStatusText");

  const MAX_SIZE_BYTES = 500 * 1024 * 1024; // 500 MB
  const ALLOWED_EXTS = ["mp4", "mov", "avi"];

  let selectedFile = null;

  // Toggle calibration input based on checkbox
  if (calibNormalCheckbox && calibSecondsInput) {
    calibNormalCheckbox.addEventListener("change", () => {
      if (calibNormalCheckbox.checked) {
        calibSecondsInput.disabled = false;
        if (calibSecondsInput.value === "0" || !calibSecondsInput.value) {
          calibSecondsInput.value = "10";
        }
        calibSecondsGroup.style.opacity = "1";
      } else {
        calibSecondsInput.value = "0";
        calibSecondsInput.disabled = true;
        calibSecondsGroup.style.opacity = "0.5";
      }
    });
  }

  // Click on dropzone opens file dialog
  dropzone.addEventListener("click", () => videoInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      videoInput.click();
    }
  });

  // Drag-and-drop event listeners
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  });

  videoInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  if (removeFileBtn) {
    removeFileBtn.addEventListener("click", () => {
      selectedFile = null;
      videoInput.value = "";
      filePreview.classList.remove("active");
      dropzone.style.display = "block";
    });
  }

  function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  function handleFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      alert(`Invalid format: .${ext}. Only MP4, MOV, and AVI videos are supported.`);
      return;
    }

    if (file.size > MAX_SIZE_BYTES) {
      alert(`File is too large (${formatBytes(file.size)}). Maximum upload size is 500 MB.`);
      return;
    }

    selectedFile = file;
    previewFilename.textContent = file.name;
    previewFilesize.textContent = formatBytes(file.size);

    filePreview.classList.add("active");
    dropzone.style.display = "none";
  }

  // Handle AJAX submission with upload progress
  uploadForm.addEventListener("submit", (e) => {
    if (!selectedFile && (!videoInput.files || videoInput.files.length === 0)) {
      e.preventDefault();
      alert("Please select or drop a CCTV video file before submitting.");
      return;
    }

    e.preventDefault();

    const formData = new FormData();
    const fileToUpload = selectedFile || videoInput.files[0];
    formData.append("video", fileToUpload);
    formData.append("calib_normal", calibNormalCheckbox.checked ? "true" : "false");
    formData.append("calib_seconds", calibSecondsInput.value || "10");

    uploadProgressContainer.style.display = "block";
    submitBtn.disabled = true;
    submitBtn.innerHTML = "<span>⏳</span> Uploading...";

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload", true);
    xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 100);
        uploadProgressBar.style.width = percent + "%";
        uploadPercentText.textContent = percent + "%";
        if (percent >= 100) {
          uploadStatusText.textContent = "Processing video on server...";
        }
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const res = JSON.parse(xhr.responseText);
          if (res.redirect_url) {
            window.location.href = res.redirect_url;
          } else if (res.job_id) {
            window.location.href = `/watch/${res.job_id}`;
          }
        } catch (err) {
          // If response was a plain redirect or text
          window.location.href = "/";
        }
      } else {
        submitBtn.disabled = false;
        submitBtn.innerHTML = "<span>▶</span> Start Analysis";
        uploadProgressContainer.style.display = "none";
        try {
          const errRes = JSON.parse(xhr.responseText);
          alert(`Upload failed: ${errRes.error || xhr.statusText}`);
        } catch (err) {
          alert(`Upload failed with status code ${xhr.status}`);
        }
      }
    };

    xhr.onerror = () => {
      submitBtn.disabled = false;
      submitBtn.innerHTML = "<span>▶</span> Start Analysis";
      uploadProgressContainer.style.display = "none";
      alert("Network error occurred during upload. Please try again.");
    };

    xhr.send(formData);
  });
});
