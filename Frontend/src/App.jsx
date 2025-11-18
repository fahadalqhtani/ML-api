import { useEffect } from "react";
import "./App.css";

// ====== Chart.js for multi-line risk chart ======
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

Chart.register(
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend
);

function App() {
  useEffect(() => {
    // ======== CONFIG ========
    const API_BASE = "https://ml-api-ec2k.onrender.com"; // عدّلها لو عندك URL مختلف
    const RISK_THRESHOLD = 85;

    const sel = document.getElementById("deviceSelect");
    const alertsList = document.getElementById("alertsList");
    const simBtn = document.getElementById("simBtn");
    const recordsContent = document.getElementById("recordsContent");

    const els = {
      temp: document.getElementById("tempVal"),
      vib: document.getElementById("vibVal"),
      pres: document.getElementById("presVal"),
      hum: document.getElementById("humVal"),
    };

    const fmt = {
      temp: (v) => `${Number(v).toFixed(1)} °C`,
      vib: (v) => `${Number(v).toFixed(2)} units`,
      pres: (v) => `${Number(v).toFixed(0)} psi`,
      hum: (v) => `${Number(v).toFixed(1)} %`,
      clock: () =>
        new Date().toLocaleTimeString("en-US", { hour12: true }),
    };

    let devices = [];
    const alertsMap = new Map();
    let simRunning = false;
    let pollTimer = null;

    // ======== Multi-line risk chart state ========
    // لكل جهاز نخزّن تاريخ الـ risk (آخر نقاط)
    const riskHistory = new Map(); // key: device name, value: array of risk values
    let multiLineChart = null;

    function pushRiskHistory(device, risk) {
      if (!riskHistory.has(device)) {
        riskHistory.set(device, []);
      }
      const arr = riskHistory.get(device);
      arr.push(risk);
      // خزن آخر 10 نقاط فقط
      if (arr.length > 10) arr.shift();
    }

    function renderMultiLineChart() {
      const canvas = document.getElementById("riskChartMulti");
      if (!canvas) return;

      const entries = Array.from(riskHistory.entries());
      if (!entries.length) {
        if (multiLineChart) {
          multiLineChart.destroy();
          multiLineChart = null;
        }
        return;
      }

      const maxLen = Math.max(...entries.map(([, arr]) => arr.length));
      if (!maxLen) return;

      const labels = Array.from({ length: maxLen }, (_, i) => `T-${maxLen - i}`);

      const datasets = entries.map(([device, arr], idx) => {
        const padded = Array(maxLen - arr.length).fill(null).concat(arr);
        const hue = (idx * 70) % 360;
        return {
          label: device,
          data: padded,
          borderColor: `hsl(${hue}, 70%, 45%)`,
          backgroundColor: `hsl(${hue}, 70%, 85%)`,
          borderWidth: 2,
          tension: 0.3,
          spanGaps: true,
          pointRadius: 3,
        };
      });

      if (multiLineChart) {
        multiLineChart.data.labels = labels;
        multiLineChart.data.datasets = datasets;
        multiLineChart.update();
        return;
      }

      multiLineChart = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
          labels,
          datasets,
        },
        options: {
          responsive: true,
          interaction: { mode: "nearest", intersect: false },
          plugins: {
            legend: {
              display: true,
              position: "bottom",
            },
            title: {
              display: false,
            },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const dev = ctx.dataset.label || "";
                  const val = ctx.parsed.y;
                  return `${dev}: ${val?.toFixed(0)}%`;
                },
              },
            },
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              ticks: {
                callback: (v) => `${v}%`,
              },
            },
          },
        },
      });
    }

    // ======== Records table ========
    function renderRecordsTable(rows) {
      if (!rows || !rows.length) {
        recordsContent.innerHTML =
          '<p style="margin:0;color:var(--muted);font-size:.9rem;">No records found for this equipment yet.</p>';
        return;
      }

      const header = `
        <thead>
          <tr>
            <th>Time</th>
            <th>Temp (°C)</th>
            <th>Vibration</th>
            <th>Pressure (psi)</th>
            <th>Humidity (%)</th>
            <th>Prediction</th>
            <th>Risk</th>
            <th>Message</th>
          </tr>
        </thead>`;

      const body = rows
        .map((r) => {
          const timeStr = new Date(r.timestamp).toLocaleString("en-US", {
            year: "numeric",
            month: "short",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: true,
          });

          return `
          <tr class="${Number(r.prediction) === 1 ? "row-fail" : ""}">
            <td>${timeStr}</td>
            <td>${Number(r.temperature).toFixed(1)}</td>
            <td>${Number(r.vibration).toFixed(2)}</td>
            <td>${Number(r.pressure).toFixed(0)}</td>
            <td>${Number(r.humidity).toFixed(1)}</td>
            <td>${Number(r.prediction) === 1 ? "Failure" : "Normal"}</td>
            <td>${(Number(r.probability) * 100).toFixed(0)}%</td>
            <td>${r.message ?? ""}</td>
          </tr>`;
        })
        .join("");

      recordsContent.innerHTML = `
        <div class="records-scroll">
          <table class="records-table">
            ${header}
            <tbody>${body}</tbody>
          </table>
        </div>`;
    }

    async function fetchRecordsForSelected() {
      const name = (sel.value || "").trim();
      if (!name || name.startsWith("Loading") || name.startsWith("Could not")) {
        recordsContent.innerHTML =
          '<p style="margin:0;color:var(--muted);font-size:.9rem;">Please select an equipment first.</p>';
        return;
      }

      try {
        const res = await fetch(
          `${API_BASE}/records?equipment_name=${encodeURIComponent(
            name
          )}&limit=50`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const { ok, data } = await res.json();
        if (ok) {
          renderRecordsTable(data || []);
        } else {
          recordsContent.innerHTML =
            '<p style="margin:0;color:var(--muted);font-size:.9rem;">Could not load records.</p>';
        }
      } catch (err) {
        console.error("records error:", err);
        recordsContent.innerHTML =
          '<p style="margin:0;color:var(--muted);font-size:.9rem;">Error loading records.</p>';
      }
    }

    // ======== Alerts UI ========
    function renderAlerts() {
      alertsList.innerHTML = "";
      const alerts = Array.from(alertsMap.values());
      if (!alerts.length) {
        const p = document.createElement("p");
        p.className = "alert-empty";
        p.textContent = "No active alerts.";
        alertsList.appendChild(p);
        return;
      }

      alerts
        .sort((a, b) => b._ts - a._ts)
        .forEach((a) => {
          const div = document.createElement("div");
          div.className = "alert-item";
          div.innerHTML = `
          <div class="alert-header">
            <span class="alert-pill">WARNING</span>
          </div>
          <p class="alert-msg">${a.message || "Abnormal condition detected."}</p>
          <span class="time">${a.time}</span>
        `;
          alertsList.appendChild(div);
        });
    }

    function upsertAlert(name, timeStr, message) {
      alertsMap.set(name, { name, time: timeStr, message, _ts: Date.now() });
      renderAlerts();
    }

    function clearAlert(name) {
      if (alertsMap.delete(name)) renderAlerts();
    }

    // ======== API helpers ========
    async function fetchLatest(name) {
      try {
        const res = await fetch(
          `${API_BASE}/latest?equipment_name=${encodeURIComponent(name)}`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const { ok, data } = await res.json();
        if (ok && data) return data;
        return null;
      } catch (err) {
        console.error("latest error:", err);
        return null;
      }
    }

    async function updateSelectedCards(source = "poll") {
      const name = (sel.value || "").trim();
      if (!name || name.startsWith("Loading") || name.startsWith("Could not"))
        return;

      const data = await fetchLatest(name);
      if (!data) return;

      els.temp.textContent = fmt.temp(data.temperature);
      els.vib.textContent = fmt.vib(data.vibration);
      els.pres.textContent = fmt.pres(data.pressure);
      els.hum.textContent = fmt.hum(data.humidity);

      const risk = Number(data.risk_score || 0);
      const prediction = Number(data.prediction || 0);
      const timeStr = new Date(data.timestamp || Date.now()).toLocaleTimeString(
        "en-US",
        { hour12: true }
      );

      // ✅ خزّن الـ risk للجهاز المختار
      pushRiskHistory(name, risk);

      if (risk >= RISK_THRESHOLD || prediction === 1) {
        upsertAlert(name, timeStr, data.message);
      } else {
        clearAlert(name);
      }

      // حدّث جدول القراءات
      await fetchRecordsForSelected();
      // وحدث الشارت بعد التحديث
      renderMultiLineChart();
    }

    // ======== Alerts refresh for all devices (بدون الجهاز المختار) ========
    async function refreshAlertsForAllDevices() {
      const selected = (sel.value || "").trim();

      for (const name of devices) {
        // تجاهل الجهاز المختار (لأنه محدث فوق في updateSelectedCards)
        if (name === selected) continue;

        const data = await fetchLatest(name);
        if (!data) {
          clearAlert(name);
          continue;
        }
        const risk = Number(data.risk_score || 0);
        const prediction = Number(data.prediction || 0);
        const timeStr = data.timestamp
          ? new Date(data.timestamp).toLocaleTimeString("en-US", {
              hour12: true,
            })
          : fmt.clock();

        // ✅ خزّن تاريخ الـ risk لكل جهاز
        pushRiskHistory(name, risk);

        if (risk >= RISK_THRESHOLD || prediction === 1) {
          upsertAlert(name, timeStr, data.message);
        } else {
          clearAlert(name);
        }
      }

      // ✅ حدّث الشارت بعد ما نحدّث كل الأجهزة
      renderMultiLineChart();
    }

    // ======== Polling ========
    function startPolling() {
      if (pollTimer) return;
      pollTimer = setInterval(async () => {
        await updateSelectedCards("poll");      // يحدّث الجهاز المختار
        await refreshAlertsForAllDevices();     // يحدّث باقي الأجهزة فقط
      }, 5000);
    }

    // ======== Simulation button ========
    if (simBtn) {
      simBtn.addEventListener("click", async () => {
        const action = simRunning ? "stop" : "start";
        simBtn.disabled = true;

        try {
          const res = await fetch(`${API_BASE}/simulation/${action}`, {
            method: "POST",
          });
          const json = await res.json().catch(() => ({}));

          if (!res.ok || json.ok === false) {
            alert(
              "Simulation request failed.\nCheck /simulation logs on Render."
            );
            return;
          }

          simRunning = !simRunning;
          simBtn.textContent = simRunning
            ? "Stop Simulation"
            : "Start Simulation";
          simBtn.style.background = simRunning ? "#dc2626" : "#2563eb";
        } catch (err) {
          console.error("Simulation toggle error:", err);
          alert(
            "Could not reach the API server.\nIs your Render service running?"
          );
        } finally {
          simBtn.disabled = false;
        }
      });
    }

    // ======== Load devices ========
    async function loadDevices() {
      const setOpt = (txt) =>
        (sel.innerHTML = `<option disabled selected>${txt}</option>`);

      for (let i = 1; i <= 5; i++) {
        try {
          const res = await fetch(`${API_BASE}/equipment`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const { ok, equipment } = await res.json();
          if (!ok) throw new Error("Backend returned ok:false");
          devices = equipment || [];
          if (!devices.length) throw new Error("No equipment in DB");

          sel.innerHTML = devices
            .map(
              (name, idx) =>
                `<option value="${name}" ${
                  idx === 0 ? "selected" : ""
                }>${name}</option>`
            )
            .join("");

          await updateSelectedCards("initial");
          await refreshAlertsForAllDevices();
          return;
        } catch (e) {
          console.warn(`[loadDevices] try ${i} failed:`, e?.message || e);
          setOpt(i === 1 ? "Warming up server…" : `Retrying… (${i}/5)`);
          await new Promise((r) => setTimeout(r, i * 1000));
        }
      }

      setOpt("Could not load devices (server waking up)…");
    }

    if (sel) {
      sel.addEventListener("change", async () => {
        els.temp.textContent =
          els.vib.textContent =
          els.pres.textContent =
          els.hum.textContent =
          "--";
        await updateSelectedCards("select-change");
      });
    }

    // === init ===
    (async () => {
      await loadDevices();
      startPolling();
    })();
  }, []);

  return (
    <>
      <div className="wrap">
        {/* Left panel */}
        <section className="panel" aria-label="Monitoring">
          <div className="header">
            <div className="title-group">
              <h1>Equipment Monitoring</h1>
              <div className="controls-row">
                <select id="deviceSelect" aria-label="Select equipment">
                  <option disabled>Loading devices…</option>
                </select>
              </div>
            </div>
            <button id="simBtn" type="button">
              Start Simulation
            </button>
          </div>

          <div className="metrics">
            <div className="metric" aria-label="Temperature">
              <h3>Temperature</h3>
              <div className="value" id="tempVal">
                --
              </div>
            </div>
            <div className="metric" aria-label="Vibration">
              <h3>Vibration</h3>
              <div className="value" id="vibVal">
                --
              </div>
            </div>
            <div className="metric" aria-label="Pressure">
              <h3>Pressure</h3>
              <div className="value" id="presVal">
                --
              </div>
            </div>
            <div className="metric" aria-label="Humidity">
              <h3>Humidity</h3>
              <div className="value" id="humVal">
                --
              </div>
            </div>
          </div>
        </section>

        {/* Right panel */}
        <aside className="alerts" aria-label="Active alerts">
          <h2>Active Alerts</h2>
          <div id="alertsList"></div>
        </aside>
      </div>

      {/* Records + chart panel */}
      <section
        id="recordsPanel"
        className="records"
        aria-label="Recorded readings"
      >
        <div className="records-header">
          <h3>Recorded Readings &amp; Failures</h3>
          <p className="records-sub">
            Shows last 50 records for the selected equipment (from Postgres /
            prediction table).
          </p>
        </div>
        <div id="recordsContent" className="records-scroll"></div>

        {/* Multi-line chart لكل الأجهزة */}
        <div className="risk-multi-wrapper">
          <h4 className="risk-multi-title">Risk Trend (All Devices)</h4>
          <canvas id="riskChartMulti" height="140"></canvas>
        </div>
      </section>
    </>
  );
}

export default App;
