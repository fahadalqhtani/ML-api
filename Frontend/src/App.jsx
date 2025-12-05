import { useEffect } from "react";
import "./App.css";

function App() {
  useEffect(() => {
    // ======== CONFIG ========
    const API_BASE = "https://ml-api-ec2k.onrender.com"; // عدّلها لو عندك URL مختلف
    const RISK_THRESHOLD = 85;

    const sel = document.getElementById("deviceSelect");
    const alertsList = document.getElementById("alertsList");
    const simBtn = document.getElementById("simBtn");
    const recordsContent = document.getElementById("recordsContent");
    const filterSel = document.getElementById("filterSelect");

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

    let allRecords = [];

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

          // class للصف حسب نوع الحالة
          const rowClass = r.sensor_error
            ? "row-sensor" // أصفر: Sensor Fault
            : Number(r.prediction) === 1
            ? "row-fail"   // أحمر: Equipment Failure
            : "";

          const predictionText = r.sensor_error
            ? "Sensor Fault"
            : Number(r.prediction) === 1
            ? "Failure"
            : "Normal";

          const riskText = r.sensor_error
            ? "--"
            : `${(Number(r.probability) * 100).toFixed(0)}%`;

          const messageText = r.sensor_error
            ? `Sensor Error: ${
                r.message || "Abnormal sensor reading detected."
              }`
            : r.message ?? "";

          return `
          <tr class="${rowClass}">
            <td>${timeStr}</td>
            <td>${Number(r.temperature).toFixed(1)}</td>
            <td>${Number(r.vibration).toFixed(2)}</td>
            <td>${Number(r.pressure).toFixed(0)}</td>
            <td>${Number(r.humidity).toFixed(1)}</td>
            <td>${predictionText}</td>
            <td>${riskText}</td>
            <td>${messageText}</td>
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

    function applyRecordsFilterAndRender() {
      if (!allRecords || !allRecords.length) {
        renderRecordsTable([]);
        return;
      }

      const mode = filterSel?.value || "all";

      let rows = allRecords;
      if (mode === "failure") {
        rows = allRecords.filter(
          (r) => !r.sensor_error && Number(r.prediction) === 1
        );
      } else if (mode === "normal") {
        rows = allRecords.filter(
          (r) => !r.sensor_error && Number(r.prediction) === 0
        );
      }
      // لو حاب تضيف فلتر للسنسور ايرور مستقبلًا تقدر تضيف خيار ثالث

      renderRecordsTable(rows);
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
          `${API_BASE}/records?equipment_name=${encodeURIComponent(name)}`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const { ok, data } = await res.json();
        if (ok) {
          allRecords = data || [];
          applyRecordsFilterAndRender();
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
            <span class="alert-pill">${a.type === "sensor" ? "SENSOR" : "WARNING"}</span>
          </div>
          <p class="alert-msg">${a.message || "Abnormal condition detected."}</p>
          <span class="time">${a.time}</span>
        `;
          alertsList.appendChild(div);
        });
    }

    function upsertAlert(name, timeStr, message, type = "failure") {
      alertsMap.set(name + ":" + type, {
        name,
        time: timeStr,
        message,
        type,
        _ts: Date.now(),
      });
      renderAlerts();
    }

    function clearAlert(name) {
      // نحذف كل أنواع التنبيهات المرتبطة بهذا الجهاز
      for (const key of Array.from(alertsMap.keys())) {
        if (key.startsWith(name + ":")) {
          alertsMap.delete(key);
        }
      }
      renderAlerts();
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
      const timeStr = new Date(
        data.timestamp || Date.now()
      ).toLocaleTimeString("en-US", { hour12: true });

      // 🔶 حالة Sensor Error
      if (data.sensor_error) {
        upsertAlert(
          name,
          timeStr,
          data.message || "Sensor error detected. Please inspect the sensor.",
          "sensor"
        );
        // ما نتعامل معها كـ Failure للمعدة
        await fetchRecordsForSelected();
        return;
      }

      // 🔴 Failure عادي
      if (risk >= RISK_THRESHOLD || prediction === 1) {
        upsertAlert(name, timeStr, data.message, "failure");
      } else {
        clearAlert(name);
      }

      await fetchRecordsForSelected();
    }

    // ======== Refresh alerts for all devices ========
    async function refreshAlertsForAllDevices() {
      const selected = (sel.value || "").trim();

      for (const name of devices) {
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

        if (data.sensor_error) {
          upsertAlert(
            name,
            timeStr,
            data.message || "Sensor error detected. Please inspect the sensor.",
            "sensor"
          );
        } else if (risk >= RISK_THRESHOLD || prediction === 1) {
          upsertAlert(name, timeStr, data.message, "failure");
        } else {
          clearAlert(name);
        }
      }
    }

    // ======== Polling ========
    function startPolling() {
      if (pollTimer) return;
      pollTimer = setInterval(async () => {
        await updateSelectedCards("poll"); // يحدّث الجهاز المختار
        await refreshAlertsForAllDevices(); // يحدّث باقي الأجهزة فقط
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

    if (filterSel) {
      filterSel.addEventListener("change", () => {
        applyRecordsFilterAndRender();
      });
    }

    // === init ===
    (async () => {
      await loadDevices();
      startPolling();
    })();

    return () => {
      if (pollTimer) clearInterval(pollTimer);
    };
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

      {/* Records panel */}
      <section
        id="recordsPanel"
        className="records"
        aria-label="Recorded readings"
      >
        <div className="records-header">
          <div>
            <h3>Recorded Readings &amp; Failures</h3>
            <p className="records-sub">
              Shows all records for the selected equipment (from Postgres /
              prediction table).
            </p>
          </div>

          <div className="records-filter">
            <label htmlFor="filterSelect" style={{ marginRight: "0.5rem" }}>
              Show:
            </label>
            <select id="filterSelect" defaultValue="all">
              <option value="all">All</option>
              <option value="failure">Failures only</option>
              <option value="normal">Normal only</option>
            </select>
          </div>
        </div>
        <div id="recordsContent" className="records-scroll"></div>
      </section>
    </>
  );
}

export default App;
