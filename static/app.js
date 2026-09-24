// static/app.js - Complete Full-Stack FDTD Frontend with Stable Node Cards & Live Telemetry

const emCanvas = document.getElementById("emCanvas");
const emCtx = emCanvas ? emCanvas.getContext("2d") : null;
const imgData = emCtx ? emCtx.createImageData(600, 600) : null;

const scopeCanvas = document.getElementById("scopeCanvas");
const scopeCtx = scopeCanvas ? scopeCanvas.getContext("2d") : null;

let latestTelemetry = null;
let isMouseDownOnCanvas = false;
let startMousePos = { x: 0, y: 0 };
let isHoveringNode = null;
let draggedNode = null;

let activeProbeType = "tx";
let activeProbeId = 0;
let activeStage = "symbols";
let lastNodeSignature = "";

const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${wsProtocol}//${location.host}/ws/sim`);
ws.binaryType = "arraybuffer";

function safeSend(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  } else {
    console.warn("WebSocket is not open yet.");
  }
}

ws.onmessage = (event) => {
  if (typeof event.data === "string") {
    const data = JSON.parse(event.data);
    latestTelemetry = data;

    const btnStart = document.getElementById("btnStart");
    const btnPause = document.getElementById("btnPause");
    if (btnStart && btnPause) {
      if (data.running) {
        btnStart.classList.add("active");
        btnPause.classList.remove("active");
      } else {
        btnPause.classList.add("active");
        btnStart.classList.remove("active");
      }
    }

    document.getElementById("statTime").textContent = `${data.time_ns.toFixed(2)} ns`;
    document.getElementById("statBits").textContent = data.bits_compared;
    document.getElementById("statErrors").textContent = data.bit_errors;
    document.getElementById("statBER").textContent = Number(data.ber).toFixed(4);
    document.getElementById("statDelay").textContent = `${data.est_delay_ns.toFixed(2)} ns`;

    renderNodeCards(data);

    const scopeModal = document.getElementById("floatingScope");
    if (scopeModal && scopeModal.style.display === "flex") {
      renderInstrumentView(activeStage);
    }
  } else {
    if (!emCtx || !imgData) return;
    const bytes = new Uint8Array(event.data);
    let p = 0;
    for (let i = 0; i < bytes.length; i++) {
      const v = bytes[i];
      if (v === 0) {
        imgData.data[p] = 30; imgData.data[p + 1] = 41; imgData.data[p + 2] = 59;
      } else {
        const diff = v - 128;
        if (diff >= 0) {
          imgData.data[p] = 255;
          imgData.data[p + 1] = Math.max(0, 255 - diff * 2);
          imgData.data[p + 2] = Math.max(0, 255 - diff * 2);
        } else {
          const neg = -diff;
          imgData.data[p] = Math.max(0, 255 - neg * 2);
          imgData.data[p + 1] = Math.max(0, 255 - neg * 2);
          imgData.data[p + 2] = 255;
        }
      }
      imgData.data[p + 3] = 255;
      p += 4;
    }
    emCtx.putImageData(imgData, 0, 0);

    drawMaterialBoundaries();
    drawNodeMarkers();
  }
};

function toPxX(val) { return (val / 10.0) * 600.0; }
function toPxY(val) { return 600.0 - (val / 10.0) * 600.0; }
function toMetersX(px) { return (px / 600.0) * 10.0; }
function toMetersY(py) { return ((600.0 - py) / 600.0) * 10.0; }

function drawMaterialBoundaries() {
  if (!latestTelemetry || !latestTelemetry.materials) return;
  emCtx.save();
  latestTelemetry.materials.forEach((mat) => {
    const x0 = toPxX(mat.x_min);
    const x1 = toPxX(mat.x_max);
    const y0 = toPxY(mat.y_max);
    const y1 = toPxY(mat.y_min);
    const angleRad = (mat.angle || 0) * Math.PI / 180;

    emCtx.save();
    emCtx.fillStyle = "rgba(245, 158, 11, 0.35)";
    emCtx.strokeStyle = "#f59e0b";
    emCtx.lineWidth = 1.8;

    if (angleRad !== 0) {
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      emCtx.translate(cx, cy);
      emCtx.rotate(angleRad);
      const w = x1 - x0;
      const h = y1 - y0;
      emCtx.fillRect(-w / 2, -h / 2, w, h);
      emCtx.strokeRect(-w / 2, -h / 2, w, h);
      emCtx.fillStyle = "#ffffff";
      emCtx.font = "bold 10px monospace";
      emCtx.fillText(mat.name.toUpperCase(), -w / 2 + 4, -h / 2 + 12);
    } else {
      emCtx.fillRect(x0, y0, x1 - x0, y1 - y0);
      emCtx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      emCtx.fillStyle = "#ffffff";
      emCtx.font = "bold 10px monospace";
      emCtx.fillText(mat.name.toUpperCase(), x0 + 4, y0 + 12);
    }
    emCtx.restore();
  });
  emCtx.restore();
}

function drawNodeMarkers() {
  if (!latestTelemetry) return;
  emCtx.save();

  function drawMarker(x, y, label, color, isHovered) {
    emCtx.beginPath();
    emCtx.arc(x, y, isHovered ? 16 : 13, 0, 2 * Math.PI);
    emCtx.fillStyle = isHovered ? `${color}dd` : `${color}88`;
    emCtx.fill();
    emCtx.strokeStyle = isHovered ? "#ffffff" : color;
    emCtx.lineWidth = isHovered ? 3.0 : 2.0;
    emCtx.stroke();
    emCtx.fillStyle = "#ffffff";
    emCtx.font = "bold 9px sans-serif";
    emCtx.textAlign = "center";
    emCtx.textBaseline = "middle";
    emCtx.fillText(label, x, y);
  }

  if (latestTelemetry.transmitters) {
    latestTelemetry.transmitters.forEach((tx) => {
      drawMarker(toPxX(tx.x), toPxY(tx.y), `TX${tx.id}`, "#ef4444", isHoveringNode === `tx_${tx.id}`);
    });
  }
  if (latestTelemetry.receivers) {
    latestTelemetry.receivers.forEach((rx) => {
      const isSel = latestTelemetry.active_rx_id === rx.id;
      drawMarker(toPxX(rx.x), toPxY(rx.y), `RX${rx.id}`, isSel ? "#38bdf8" : "#0284c7", isHoveringNode === `rx_${rx.id}`);
    });
  }
  emCtx.restore();
}

function renderNodeCards(data) {
  const container = document.getElementById("nodeCardsContainer");
  if (!container) return;

  // Prevent re-rendering while a dropdown menu is open/focused
  if (document.activeElement && document.activeElement.classList.contains("link-select")) {
    return;
  }

  const txSigs = data.transmitters ? data.transmitters.map(t => `${t.id}:${t.fc}`).join(",") : "";
  const rxSigs = data.receivers ? data.receivers.map(r => `${r.id}:${r.fc}:${r.linked_tx}:${data.active_rx_id === r.id}`).join(",") : "";
  const signature = `${txSigs}|${rxSigs}`;

  if (signature === lastNodeSignature && container.children.length > 0) return;
  lastNodeSignature = signature;

  container.innerHTML = "";

  if (data.transmitters) {
    data.transmitters.forEach((tx) => {
      const card = document.createElement("div");
      card.className = "node-card";
      card.style.cssText = "background:#131b29; border:1.5px solid #ef444488; border-radius:6px; padding:8px 10px; cursor:pointer; display:flex; flex-direction:column; gap:4px; transition:all 0.15s;";
      card.innerHTML = `
        <div style="color:#ef4444; display:flex; justify-content:space-between; align-items:center; width:100%; font-size:11.5px; font-weight:700;">
          <span>Transmitter ${tx.id}</span>
          <span style="font-size:10px; background:#ef444422; border:1px solid #ef444455; color:#fca5a5; padding:2px 8px; border-radius:3px; font-weight:600;">Probe ~</span>
        </div>
        <div style="font-size:10.5px; color:#94a3b8;">${(tx.fc / 1e9).toFixed(2)} GHz | ${(tx.rb / 1e6).toFixed(0)} Mbps</div>
      `;
      card.onclick = () => setProbe("tx", tx.id);
      container.appendChild(card);
    });
  }

  if (data.receivers) {
    data.receivers.forEach((rx) => {
      const isSelected = data.active_rx_id === rx.id;
      const card = document.createElement("div");
      card.className = "node-card";
      card.style.cssText = `background:#131b29; border:1.5px solid ${isSelected ? "#38bdf8" : "#0ea5e966"}; border-radius:6px; padding:8px 10px; display:flex; flex-direction:column; gap:6px; transition:all 0.15s;`;
      
      let txOptions = "";
      if (data.transmitters) {
        data.transmitters.forEach(t => {
          const sel = (rx.linked_tx === t.id) ? "selected" : "";
          txOptions += `<option value="${t.id}" ${sel}>TX ${t.id}</option>`;
        });
      }

      card.innerHTML = `
        <div style="color:#0ea5e9; display:flex; justify-content:space-between; align-items:center; width:100%; font-size:11.5px; font-weight:700; cursor:pointer;" onclick="selectReceiver(${rx.id})">
          <span>Receiver ${rx.id} ${isSelected ? "(Active)" : ""}</span>
          <span style="font-size:10px; background:#0ea5e922; border:1px solid #0ea5e955; color:#7dd3fc; padding:2px 8px; border-radius:3px; font-weight:600;">Probe ~</span>
        </div>
        <div style="font-size:10.5px; color:#94a3b8; display:flex; justify-content:space-between; align-items:center;">
          <span>Link Source:</span>
          <select class="preset-select link-select" data-rx="${rx.id}" style="padding:2px 4px; font-size:10px;">${txOptions}</select>
        </div>
        <div style="font-size:10.5px; color:#94a3b8; display:flex; justify-content:space-between;">
          <span>BER:</span>
          <span style="color:#38bdf8; font-family:monospace; font-weight:bold;">${Number(rx.ber).toFixed(4)}</span>
        </div>
        <div style="font-size:10.5px; color:#94a3b8; display:flex; justify-content:space-between;">
          <span>Delay:</span>
          <span style="color:#f8fafc; font-family:monospace;">${rx.delay_ns.toFixed(2)} ns</span>
        </div>
      `;

      card.querySelector(".link-select").onchange = (e) => {
        e.stopPropagation();
        safeSend({ type: "set_rx_link", rx_id: rx.id, tx_id: parseInt(e.target.value) });
      };

      container.appendChild(card);
    });
  }
}

function selectReceiver(rxId) {
  safeSend({ type: "select_receiver", rx_id: rxId });
  setProbe("rx", rxId);
}

function setProbe(type, id) {
  activeProbeType = type;
  activeProbeId = id;

  const conf = type === "tx" ? {
    title: `Probe: Transmitter ${id}`,
    ledColor: "#ef4444",
    stages: [
      { key: "symbols", label: "Bits", color: "#f59e0b", type: "step" },
      { key: "shaped", label: "RRC Baseband", color: "#38bdf8", type: "wave" },
      { key: "bpsk", label: "BPSK RF", color: "#60a5fa", type: "wave" },
      { key: "spectrum", label: "FFT Spectrum", color: "#facc15", type: "fft" }
    ]
  } : {
    title: `Probe: Receiver ${id}`,
    ledColor: "#0ea5e9",
    stages: [
      { key: "rx_raw", label: "Antenna RF", color: "#34d399", type: "wave" },
      { key: "rx_bpf", label: "BP Filter", color: "#10b981", type: "wave" },
      { key: "rx_mixed", label: "Mixer Out", color: "#2dd4bf", type: "wave" },
      { key: "rx_matched", label: "Matched Filter", color: "#c084fc", type: "wave" },
      { key: "eye_diagram", label: "Eye Diagram", color: "#e879f9", type: "eye" },
      { key: "spectrum", label: "FFT Spectrum", color: "#facc15", type: "fft" }
    ]
  };

  document.getElementById("scopeTitleText").textContent = conf.title;
  document.getElementById("scopeProbeLed").style.background = conf.ledColor;

  const tabsContainer = document.getElementById("scopeTabsContainer");
  tabsContainer.innerHTML = "";

  conf.stages.forEach((stg, index) => {
    const tab = document.createElement("div");
    tab.className = `scope-tab ${index === 0 ? "active" : ""}`;
    tab.textContent = stg.label;
    tab.onclick = () => {
      Array.from(tabsContainer.children).forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      activeStage = stg.key;
      document.getElementById("scopeChannelName").textContent = `Channel: ${stg.label}`;
      renderInstrumentView(activeStage);
    };
    tabsContainer.appendChild(tab);
  });

  activeStage = conf.stages[0].key;
  document.getElementById("scopeChannelName").textContent = `Channel: ${conf.stages[0].label}`;
  
  const scopeModal = document.getElementById("floatingScope");
  if (scopeModal) {
    scopeModal.style.display = "flex";
  }

  renderInstrumentView(activeStage);
}

function renderInstrumentView(stageKey) {
  if (!latestTelemetry || !scopeCtx) return;
  const w = scopeCtx.canvas.width;
  const h = scopeCtx.canvas.height;
  scopeCtx.clearRect(0, 0, w, h);

  scopeCtx.strokeStyle = "#162032";
  scopeCtx.lineWidth = 1;
  scopeCtx.beginPath();
  scopeCtx.moveTo(0, h / 2); scopeCtx.lineTo(w, h / 2);
  scopeCtx.stroke();

  if (activeProbeType === "tx") {
    const tx = latestTelemetry.transmitters.find(t => t.id === activeProbeId);
    if (!tx) return;

    if (stageKey === "spectrum") {
      const spectrum = tx.spectrum || [];
      const n = spectrum.length;
      if (n === 0) return;
      scopeCtx.fillStyle = "#facc15";
      const barW = w / n;
      for (let i = 0; i < n; i++) {
        const barH = spectrum[i] * (h - 20);
        scopeCtx.fillRect(i * barW, h - barH, Math.max(1, barW - 1), barH);
      }
      return;
    }

    const values = tx[stageKey] || [];
    if (values.length < 2) return;
    drawWaveform(values, stageKey === "symbols" ? "step" : "wave", "#f59e0b");
    return;
  }

  if (stageKey === "eye_diagram") {
    const values = latestTelemetry.rx_matched || [];
    if (values.length < 32) return;
    const symLen = 16;
    scopeCtx.strokeStyle = "rgba(232, 121, 249, 0.45)";
    scopeCtx.lineWidth = 1.5;
    for (let s = 0; s < values.length - symLen * 2; s += symLen) {
      scopeCtx.beginPath();
      for (let i = 0; i < symLen * 2; i++) {
        const x = (i / (symLen * 2 - 1)) * w;
        const normY = Math.max(0, Math.min(1, (values[s + i] + 1.8) / 3.6));
        const y = h - normY * h;
        if (i === 0) scopeCtx.moveTo(x, y);
        else scopeCtx.lineTo(x, y);
      }
      scopeCtx.stroke();
    }
    return;
  }

  if (stageKey === "spectrum") {
    const spectrum = latestTelemetry.spectrum || [];
    const n = spectrum.length;
    if (n === 0) return;
    scopeCtx.fillStyle = "#facc15";
    const barW = w / n;
    for (let i = 0; i < n; i++) {
      const barH = spectrum[i] * (h - 20);
      scopeCtx.fillRect(i * barW, h - barH, Math.max(1, barW - 1), barH);
    }
    return;
  }

  const values = latestTelemetry[stageKey] || [];
  if (values.length < 2) return;
  drawWaveform(values, "wave", "#38bdf8");
}

function drawWaveform(values, type, color) {
  const w = scopeCtx.canvas.width;
  const h = scopeCtx.canvas.height;
  let maxAbs = 0.05;
  for (let i = 0; i < values.length; i++) {
    if (Math.abs(values[i]) > maxAbs) maxAbs = Math.abs(values[i]);
  }
  const yMax = maxAbs * 1.25;
  const yMin = -yMax;

  scopeCtx.strokeStyle = color;
  scopeCtx.lineWidth = 2.0;
  scopeCtx.beginPath();
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * w;
    const normY = Math.max(0, Math.min(1, (values[i] - yMin) / (yMax - yMin)));
    const y = h - normY * h;
    if (i === 0) scopeCtx.moveTo(x, y);
    else {
      if (type === "step") {
        const prevY = h - Math.max(0, Math.min(1, (values[i - 1] - yMin) / (yMax - yMin))) * h;
        scopeCtx.lineTo(x, prevY);
      }
      scopeCtx.lineTo(x, y);
    }
  }
  scopeCtx.stroke();
}

if (emCanvas) {
  emCanvas.addEventListener("click", (e) => {
    if (!latestTelemetry) return;
    const rect = emCanvas.getBoundingClientRect();
    const cx = toMetersX(((e.clientX - rect.left) / rect.width) * 600.0);
    const cy = toMetersY(((e.clientY - rect.top) / rect.height) * 600.0);

    if (latestTelemetry.transmitters) {
      for (const tx of latestTelemetry.transmitters) {
        if (Math.hypot(cx - tx.x, cy - tx.y) < 0.7) {
          openNodeParameterPopup("tx", tx.id, e.clientX, e.clientY);
          return;
        }
      }
    }
    if (latestTelemetry.receivers) {
      for (const rx of latestTelemetry.receivers) {
        if (Math.hypot(cx - rx.x, cy - rx.y) < 0.7) {
          openNodeParameterPopup("rx", rx.id, e.clientX, e.clientY);
          return;
        }
      }
    }
  });

  emCanvas.addEventListener("mousedown", (e) => {
    if (!latestTelemetry) return;
    const rect = emCanvas.getBoundingClientRect();
    const cx = toMetersX(((e.clientX - rect.left) / rect.width) * 600.0);
    const cy = toMetersY(((e.clientY - rect.top) / rect.height) * 600.0);

    startMousePos = { x: e.clientX, y: e.clientY };

    if (latestTelemetry.transmitters) {
      for (const tx of latestTelemetry.transmitters) {
        if (Math.hypot(cx - tx.x, cy - tx.y) < 0.6) {
          draggedNode = { type: "tx", id: tx.id };
          isMouseDownOnCanvas = true;
          return;
        }
      }
    }
    if (latestTelemetry.receivers) {
      for (const rx of latestTelemetry.receivers) {
        if (Math.hypot(cx - rx.x, cy - rx.y) < 0.6) {
          draggedNode = { type: "rx", id: rx.id };
          selectReceiver(rx.id);
          isMouseDownOnCanvas = true;
          return;
        }
      }
    }
    isMouseDownOnCanvas = true;
  });

  emCanvas.addEventListener("mousemove", (e) => {
    const rect = emCanvas.getBoundingClientRect();
    const cx = toMetersX(((e.clientX - rect.left) / rect.width) * 600.0);
    const cy = toMetersY(((e.clientY - rect.top) / rect.height) * 600.0);
    document.getElementById("canvasCoordsTelemetry").textContent = `Coords: (${cx.toFixed(2)} m, ${cy.toFixed(2)} m)`;

    if (isMouseDownOnCanvas && draggedNode) {
      const d = Math.hypot(e.clientX - startMousePos.x, e.clientY - startMousePos.y);
      if (d > 5) {
        const posX = Math.max(0.5, Math.min(9.5, cx));
        const posY = Math.max(0.5, Math.min(9.5, cy));

        if (draggedNode.type === "tx") {
          safeSend({ type: "move_tx", tx_id: draggedNode.id, x: posX, y: posY });
        } else if (draggedNode.type === "rx") {
          safeSend({ type: "move_rx", rx_id: draggedNode.id, x: posX, y: posY });
        }
      }
    }
  });

  window.addEventListener("mouseup", () => {
    isMouseDownOnCanvas = false;
    draggedNode = null;
  });
}

function openNodeParameterPopup(type, id, mouseX, mouseY) {
  let existing = document.getElementById("nodeParamPopup");
  if (existing) existing.remove();

  let nodeData = null;
  if (type === "tx" && latestTelemetry.transmitters) {
    nodeData = latestTelemetry.transmitters.find(t => t.id === id);
  } else if (type === "rx" && latestTelemetry.receivers) {
    nodeData = latestTelemetry.receivers.find(r => r.id === id);
  }
  if (!nodeData) return;

  const popup = document.createElement("div");
  popup.id = "nodeParamPopup";
  popup.className = "draggable-modal";
  popup.style.cssText = `top: ${Math.min(window.innerHeight - 340, mouseY)}px; left: ${Math.min(window.innerWidth - 320, mouseX)}px; width: 310px; display: flex; border: 1.5px solid ${type === "tx" ? "#ef4444" : "#38bdf8"};`;

  const title = type === "tx" ? `Transmitter ${id} Details & Controls` : `Receiver ${id} Details & Controls`;
  const headerColor = type === "tx" ? "#ef4444" : "#0ea5e9";

  const currentFc = (nodeData.fc / 1e9).toFixed(2);
  const currentRb = (nodeData.rb / 1e6).toFixed(0);
  const currentAmp = type === "tx" ? (nodeData.amp || 2.0).toFixed(1) : null;

  let detailsHTML = `
    <div style="display: flex; flex-direction: column; gap: 8px; background: #090d15; padding: 8px; border-radius: 4px; border: 1px solid #1e293b;">
      <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:11px;">
        <span>Position (X, Y):</span>
        <span style="color:#f8fafc; font-family:monospace;">(${nodeData.x.toFixed(2)}m, ${nodeData.y.toFixed(2)}m)</span>
      </div>
  `;

  if (type === "rx") {
    detailsHTML += `
      <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:11px;">
        <span>Bit Error Rate (BER):</span>
        <span style="color:#38bdf8; font-family:monospace; font-weight:bold;">${Number(nodeData.ber).toFixed(4)}</span>
      </div>
      <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:11px;">
        <span>Est. Propagation Delay:</span>
        <span style="color:#f8fafc; font-family:monospace;">${nodeData.delay_ns.toFixed(2)} ns</span>
      </div>
      <div style="display:flex; justify-content:space-between; color:#94a3b8; font-size:11px;">
        <span>Bits Compared / Errors:</span>
        <span style="color:#f8fafc; font-family:monospace;">${nodeData.bits_compared} / ${nodeData.bit_errors}</span>
      </div>
    `;
  }

  detailsHTML += `</div>`;

  let slidersHTML = `
    <div style="display: flex; flex-direction: column; gap: 8px;">
      <label style="font-size: 11px; color: #94a3b8;">
        ${type === "tx" ? "Carrier" : "Tuned"} Freq: <span id="valFc" style="color: #f8fafc; font-weight: bold;">${currentFc}</span> GHz
        <input type="range" id="rangeFc" min="0.5" max="3.0" step="0.1" value="${currentFc}" style="width:100%; accent-color:${headerColor};">
      </label>
      <label style="font-size: 11px; color: #94a3b8;">
        Bit Rate: <span id="valRb" style="color: #f8fafc; font-weight: bold;">${currentRb}</span> Mbps
        <input type="range" id="rangeRb" min="50" max="1000" step="50" value="${currentRb}" style="width:100%; accent-color:${headerColor};">
      </label>
  `;

  if (type === "tx") {
    slidersHTML += `
      <label style="font-size: 11px; color: #94a3b8;">
        Amplitude: <span id="valAmp" style="color: #f8fafc; font-weight: bold;">${currentAmp}</span> V
        <input type="range" id="rangeAmp" min="0.5" max="5.0" step="0.5" value="${currentAmp}" style="width:100%; accent-color:${headerColor};">
      </label>
    `;
  }

  slidersHTML += `</div>`;

  popup.innerHTML = `
    <div class="modal-header" id="paramHeader">
      <div class="modal-title" style="color:${headerColor};">
        <span>${title}</span>
      </div>
      <button class="icon-btn" id="btnClosePopup">&times;</button>
    </div>
    <div style="padding: 12px; display: flex; flex-direction: column; gap: 10px; font-size: 11.5px;">
      ${detailsHTML}
      ${slidersHTML}
      <div style="display:flex; justify-content:flex-end; align-items:center; border-top: 1px solid #1e293b; padding-top: 8px;">
        <button id="btnTriggerProbe" style="background:${headerColor}22; border:1px solid ${headerColor}; color:${headerColor}; padding:5px 14px; border-radius:4px; cursor:pointer; font-weight:600;">Probe ${type.toUpperCase()} ${id}</button>
      </div>
    </div>
  `;

  document.body.appendChild(popup);
  makeDraggable("nodeParamPopup", "paramHeader");

  popup.querySelector("#btnClosePopup").onclick = () => popup.remove();
  
  popup.querySelector("#btnTriggerProbe").onclick = () => {
    if (type === "rx") {
      selectReceiver(id);
    }
    setProbe(type, id);
  };

  const rangeFc = popup.querySelector("#rangeFc");
  const valFc = popup.querySelector("#valFc");
  rangeFc.oninput = (e) => {
    valFc.textContent = e.target.value;
    const newFc = parseFloat(e.target.value) * 1e9;
    if (type === "tx") {
      safeSend({ type: "set_tx_params", tx_id: id, fc: newFc });
    } else {
      safeSend({ type: "set_rx_params", rx_id: id, fc: newFc });
    }
  };

  const rangeRb = popup.querySelector("#rangeRb");
  const valRb = popup.querySelector("#valRb");
  rangeRb.oninput = (e) => {
    valRb.textContent = e.target.value;
    const newRb = parseFloat(e.target.value) * 1e6;
    if (type === "tx") {
      safeSend({ type: "set_tx_params", tx_id: id, rb: newRb });
    } else {
      safeSend({ type: "set_rx_params", rx_id: id, rb: newRb });
    }
  };

  if (type === "tx") {
    const rangeAmp = popup.querySelector("#rangeAmp");
    const valAmp = popup.querySelector("#valAmp");
    rangeAmp.oninput = (e) => {
      valAmp.textContent = e.target.value;
      safeSend({ type: "set_tx_params", tx_id: id, amp: parseFloat(e.target.value) });
    };
  }
}

function makeDraggable(modalId, handleId) {
  const modal = document.getElementById(modalId);
  const handle = document.getElementById(handleId);
  if (!modal || !handle) return;
  let isDragging = false, offX = 0, offY = 0;
  handle.addEventListener("mousedown", (e) => {
    isDragging = true;
    offX = e.clientX - modal.offsetLeft;
    offY = e.clientY - modal.offsetTop;
  });
  window.addEventListener("mouseup", () => isDragging = false);
  window.addEventListener("mousemove", (e) => {
    if (isDragging) {
      modal.style.left = `${e.clientX - offX}px`;
      modal.style.top = `${e.clientY - offY}px`;
    }
  });
}
makeDraggable("floatingScope", "scopeHeader");

document.getElementById("btnCloseScope").onclick = () => document.getElementById("floatingScope").style.display = "none";
document.getElementById("btnStart").onclick = () => safeSend({ type: "start" });
document.getElementById("btnPause").onclick = () => safeSend({ type: "pause" });
document.getElementById("btnReset").onclick = () => safeSend({ type: "reset" });
document.getElementById("btnClearWalls").onclick = () => safeSend({ type: "clear_walls" });
const btnClearWallsAlt = document.getElementById("btnClearWallsAlt");
if (btnClearWallsAlt) btnClearWallsAlt.onclick = () => safeSend({ type: "clear_walls" });

document.getElementById("btnAddTransmitterBtn").onclick = () => {
  const n = latestTelemetry && latestTelemetry.transmitters ? latestTelemetry.transmitters.length : 1;
  safeSend({ type: "add_transmitter", x: 1.5, y: Math.min(8.5, 2.0 + n * 1.5) });
};

document.getElementById("btnAddReceiverBtn").onclick = () => {
  const n = latestTelemetry && latestTelemetry.receivers ? latestTelemetry.receivers.length : 1;
  safeSend({ type: "add_receiver", x: 8.5, y: Math.min(8.5, 2.0 + n * 1.5) });
};

const materialSelect = document.getElementById("materialSelect");
const customNameField = document.getElementById("customNameField");

if (materialSelect) {
  materialSelect.onchange = (e) => {
    if (e.target.value === "custom") {
      customNameField.style.display = "flex";
    } else {
      customNameField.style.display = "none";
    }
  };
}

document.getElementById("btnAddSelectedMaterial").onclick = () => {
  const selectedValue = materialSelect.value;
  const name = selectedValue === "custom" 
    ? (document.getElementById("customMatName").value.trim() || "custom") 
    : selectedValue;

  let x_min = parseFloat(document.getElementById("customXMin").value);
  let x_max = parseFloat(document.getElementById("customXMax").value);
  let y_min = parseFloat(document.getElementById("customYMin").value);
  let y_max = parseFloat(document.getElementById("customYMax").value);
  const angle = parseFloat(document.getElementById("customAngle").value) || 0;

  if (x_min >= x_max) {
    const temp = x_min;
    x_min = x_max - 0.2;
    x_max = temp + 0.2;
  }

  if (y_min >= y_max) {
    const temp = y_min;
    y_min = y_max - 0.2;
    y_max = temp + 0.2;
  }

  const payload = {
    type: "add_material",
    name: name,
    x_min: x_min,
    x_max: x_max,
    y_min: y_min,
    y_max: y_max,
    angle: angle,
    relative_permittivity: parseFloat(document.getElementById("customPermittivity")?.value) || 15.0,
    relative_permeability: parseFloat(document.getElementById("customPermeability")?.value) || 1.0,
    conductivity: parseFloat(document.getElementById("customConductivity")?.value) || 0.0
  };

  safeSend(payload);
};

document.getElementById("scenarioSelect").onchange = (e) => {
  const val = e.target.value;
  if (val === "los") {
    safeSend({ type: "clear_walls" });
  } else if (val === "concrete_wall") {
    safeSend({ type: "add_material", name: "concrete", x_min: 4.8, x_max: 5.2, y_min: 2.0, y_max: 8.0 });
  }
};