// watch.js - Real-time Analysis Telemetry, Live Chart.js & Alert Merging Dashboard

document.addEventListener("DOMContentLoaded", () => {
  const jobId = window.JOB_ID;
  if (!jobId) return;

  // DOM Elements
  const statusBanner = document.getElementById("statusBanner");
  const statusIcon = document.getElementById("statusIcon");
  const statusTitle = document.getElementById("statusTitle");
  const statusSub = document.getElementById("statusSub");

  const densityBadge = document.getElementById("densityBadge");
  const densityText = document.getElementById("densityText");
  const densityConf = document.getElementById("densityConf");

  const curScore = document.getElementById("curScore");
  const curThreshold = document.getElementById("curThreshold");
  const curTime = document.getElementById("curTime");
  const curFrame = document.getElementById("curFrame");
  const curBoxes = document.getElementById("curBoxes");

  const jobProgressBar = document.getElementById("jobProgressBar");
  const jobProgressPercent = document.getElementById("jobProgressPercent");
  const jobStatusText = document.getElementById("jobStatusText");

  const alertLogBody = document.getElementById("alertLogBody");
  const alertCountBadge = document.getElementById("alertCountBadge");

  const summaryCard = document.getElementById("summaryCard");
  const sumAlertEvents = document.getElementById("sumAlertEvents");
  const sumAlertFramesPct = document.getElementById("sumAlertFramesPct");
  const sumTotalFrames = document.getElementById("sumTotalFrames");
  const summaryDuration = document.getElementById("summaryDuration");
  const distBarSparse = document.getElementById("distBarSparse");
  const distBarMedium = document.getElementById("distBarMedium");
  const distBarDense = document.getElementById("distBarDense");
  const distSummaryLabel = document.getElementById("distSummaryLabel");

  // Initialize Chart.js
  const ctx = document.getElementById("anomalyChart").getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, 240);
  gradient.addColorStop(0, "rgba(6, 182, 212, 0.35)");
  gradient.addColorStop(1, "rgba(6, 182, 212, 0.0)");

  const anomalyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Anomaly Score",
          data: [],
          borderColor: "#06b6d4",
          backgroundColor: gradient,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.25,
          fill: true
        },
        {
          label: "Safety Threshold",
          data: [],
          borderColor: "#ef4444",
          borderWidth: 1.5,
          borderDash: [5, 4],
          pointRadius: 0,
          pointHoverRadius: 0,
          tension: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          labels: {
            color: "#9ca3af",
            font: { size: 11, family: "-apple-system, BlinkMacSystemFont, Segoe UI" }
          }
        },
        tooltip: {
          backgroundColor: "rgba(17, 24, 39, 0.95)",
          titleColor: "#f3f4f6",
          bodyColor: "#d1d5db",
          borderColor: "rgba(75, 85, 99, 0.5)",
          borderWidth: 1
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Time (seconds)",
            color: "#6b7280",
            font: { size: 10 }
          },
          ticks: {
            color: "#6b7280",
            maxTicksLimit: 10,
            font: { size: 10 }
          },
          grid: {
            color: "rgba(75, 85, 99, 0.15)"
          }
        },
        y: {
          title: {
            display: true,
            text: "Normalized Score",
            color: "#6b7280",
            font: { size: 10 }
          },
          ticks: {
            color: "#6b7280",
            font: { size: 10 }
          },
          grid: {
            color: "rgba(75, 85, 99, 0.15)"
          },
          suggestedMin: 0,
          suggestedMax: 0.05
        }
      }
    }
  });

  let isDone = false;
  let pollingInterval = null;

  async function fetchStats() {
    try {
      const response = await fetch(`/stats/${jobId}`);
      if (!response.ok) return;

      const data = await response.json();

      // Update Progress
      const pct = Math.round(data.progress_percent || 0);
      jobProgressBar.style.width = pct + "%";
      jobProgressPercent.textContent = pct + "%";

      if (data.status === "queued") {
        jobStatusText.textContent = "Queued for processing...";
      } else if (data.status === "running") {
        jobStatusText.textContent = `Analyzing frames (${pct}%)...`;
      } else if (data.status === "done") {
        jobStatusText.textContent = "Analysis complete";
      } else if (data.status === "error") {
        jobStatusText.textContent = `Error: ${data.error_message || "Unknown error"}`;
        statusBanner.className = "status-banner alert";
        statusTitle.textContent = "PROCESSING ERROR";
        statusSub.textContent = data.error_message || "Encountered an issue during video processing.";
        clearInterval(pollingInterval);
        return;
      }

      // Update Latest Telemetry
      if (data.latest) {
        const l = data.latest;

        // Density Badge: green sparse, orange medium, red dense
        const density = (l.density || "sparse").toLowerCase();
        densityBadge.className = `density-badge density-${density}`;
        densityText.textContent = density.toUpperCase();
        densityConf.textContent = `(${Math.round((l.density_conf || 0) * 100)}%)`;

        // Scores & Time
        curScore.textContent = Number(l.score || 0).toFixed(4);
        curThreshold.textContent = Number(l.threshold || 0).toFixed(4);
        curTime.textContent = `${Number(l.time || 0).toFixed(1)}s`;
        curFrame.textContent = l.frame || 0;
        curBoxes.textContent = l.boxes || 0;

        // Anomaly Banner
        if (l.alert) {
          statusBanner.className = "status-banner alert";
          statusIcon.textContent = "🚨";
          statusTitle.textContent = "ANOMALY DETECTED";
          statusSub.textContent = `Reconstruction error (${Number(l.score).toFixed(4)}) exceeds adaptive safety threshold (${Number(l.threshold).toFixed(4)})`;
        } else {
          statusBanner.className = "status-banner normal";
          statusIcon.textContent = "🟢";
          statusTitle.textContent = "SCENE NORMAL";
          statusSub.textContent = "Next-frame predictor error within normal bounds";
        }
      }

      // Update Chart Data Points
      if (data.chart_points && data.chart_points.length > 0) {
        const labels = data.chart_points.map(p => p.time.toFixed(1));
        const scores = data.chart_points.map(p => p.score);
        const thresholds = data.chart_points.map(p => p.threshold);

        anomalyChart.data.labels = labels;
        anomalyChart.data.datasets[0].data = scores;
        anomalyChart.data.datasets[1].data = thresholds;
        anomalyChart.update("none"); // Update without laggy animations
      }

      // Update Alert Log Table
      if (data.alert_events && data.alert_events.length > 0) {
        alertCountBadge.textContent = `${data.alert_events.length} Event${data.alert_events.length === 1 ? '' : 's'}`;
        alertLogBody.innerHTML = data.alert_events.map((ev, idx) => {
          const duration = Math.max(0.1, ev.end_time - ev.start_time).toFixed(1);
          const den = (ev.density || "sparse").toLowerCase();
          return `
            <tr>
              <td style="color: var(--accent-cyan);">#${idx + 1}</td>
              <td>${ev.start_time.toFixed(1)}s</td>
              <td>${ev.end_time.toFixed(1)}s <span style="font-size:0.75rem; color:var(--text-dim);">(${duration}s)</span></td>
              <td style="color: #ef4444; font-weight: 700;">${Number(ev.peak_score).toFixed(4)}</td>
              <td><span class="density-badge density-${den}" style="font-size:0.7rem; padding: 0.15rem 0.5rem;">${den.toUpperCase()}</span></td>
            </tr>
          `;
        }).join("");
      } else {
        alertCountBadge.textContent = "0 Events";
      }

      // Handle Job Completion Summary
      if (data.status === "done" && !isDone) {
        isDone = true;
        clearInterval(pollingInterval);

        if (data.summary) {
          const s = data.summary;
          sumAlertEvents.textContent = s.alert_events_count;
          sumAlertFramesPct.textContent = `${s.alert_frames_percent}%`;
          sumTotalFrames.textContent = s.total_frames;
          summaryDuration.textContent = `Total: ${Number(s.duration_seconds || 0).toFixed(1)}s`;

          const pSparse = s.density_percentages?.sparse || 0;
          const pMedium = s.density_percentages?.medium || 0;
          const pDense = s.density_percentages?.dense || 0;

          distBarSparse.style.width = pSparse + "%";
          distBarMedium.style.width = pMedium + "%";
          distBarDense.style.width = pDense + "%";

          distSummaryLabel.textContent = `Sparse ${pSparse}% | Medium ${pMedium}% | Dense ${pDense}%`;
        }

        summaryCard.classList.add("active");
        statusBanner.className = "status-banner normal";
        statusIcon.textContent = "🏁";
        statusTitle.textContent = "ANALYSIS COMPLETE";
        statusSub.textContent = "Full video processed. Summary and export packages ready below.";
      }

    } catch (err) {
      console.error("Telemetry poll failed:", err);
    }
  }

  // Poll stats every 800ms
  fetchStats();
  pollingInterval = setInterval(fetchStats, 800);
});
