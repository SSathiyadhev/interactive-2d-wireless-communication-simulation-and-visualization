// static/app.js - Complete FDTD Workbench Frontend

const emCanvas = document.getElementById("emCanvas");
const emCtx = emCanvas ? emCanvas.getContext("2d") : null;

// Initial telemetry fallback so nodes display instantly on load
let latestTelemetry = {
  transmitters: [{ id: 0, x: 2.0, y: 7.0, fc: 1e9, rb: 500e6, amp: 2.0 }],
  receivers: [{ id: 0, x: 8.0, y: 5.0, fc: 1e9, rb: 500e6 }],
  observation_points: [{ id: 0, x: 5.0, y: 5.0, label: "Grid Probe 0" }],
  materials: []
};

let fieldWidth = 1000;
let fieldHeight = 1000;
let imgData = emCtx ? emCtx.createImageData(fieldWidth, fieldHeight) : null;

let isMouseDownOnCanvas = false;
let startMousePos = { x: 0, y: 0 };
let draggedNode = null;
let activeModalZIndex = 1000;

// Box Drawing State
let isDrawingBox = false;
let boxStartCoord = { x: 0, y: 0 };
let boxCurrentCoord = { x: 0, y: 0 };

// Custom Material Configuration State
let customMaterialConfig = {
  name: "composite",
  permittivity: 10.0,
  permeability: 1.0,
  conductivity: 0.0
};

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

    if (data.grid_width && data.grid_height) {
      if (data.grid_width !== fieldWidth || data.grid_height !== fieldHeight) {
        fieldWidth = data.grid_width;
        fieldHeight = data.grid_height;
        if (emCtx) {
          imgData = emCtx.createImageData(fieldWidth, fieldHeight);
          emCanvas.width = fieldWidth;
          emCanvas.height = fieldHeight;
        }
      }
    }

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

    renderLinkEvaluators(data);
    refreshOpenModals();
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
    drawActiveDragBox();
  }
};

const SIM_WIDTH = 10.0;
const SIM_HEIGHT = 10.0;

function toPxX(x) { return (x / SIM_WIDTH) * fieldWidth; }
function toPxY(y) { return fieldHeight - (y / SIM_HEIGHT) * fieldHeight; }
function toMetersX(px) { return (px / fieldWidth) * SIM_WIDTH; }
function toMetersY(py) { return ((fieldHeight - py) / fieldHeight) * SIM_HEIGHT; }

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
    emCtx.fillStyle = "rgba(245, 158, 11, 0.4)";
    emCtx.strokeStyle = "#f59e0b";
    emCtx.lineWidth = 1.5;

    if (angleRad !== 0) {
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      emCtx.translate(cx, cy);
      emCtx.rotate(angleRad);
      const w = x1 - x0;
      const h = y1 - y0;
      emCtx.fillRect(-w / 2, -h / 2, w, h);
      emCtx.strokeRect(-w / 2, -h / 2, w, h);
    } else {
      emCtx.fillRect(x0, y0, x1 - x0, y1 - y0);
      emCtx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      emCtx.fillStyle = "#ffffff";
      emCtx.font = "bold 9px monospace";
      emCtx.fillText((mat.name || "mat").toUpperCase(), x0 + 4, y0 + 12);
    }
    emCtx.restore();
  });
  emCtx.restore();
}

function drawActiveDragBox() {
  if (!isDrawingBox) return;
  emCtx.save();
  emCtx.strokeStyle = "#38bdf8";
  emCtx.lineWidth = 2;
  emCtx.setLineDash([5, 5]);
  emCtx.fillStyle = "rgba(56, 189, 248, 0.25)";

  const px0 = toPxX(boxStartCoord.x);
  const py0 = toPxY(boxStartCoord.y);
  const px1 = toPxX(boxCurrentCoord.x);
  const py1 = toPxY(boxCurrentCoord.y);

  const x = Math.min(px0, px1);
  const y = Math.min(py0, py1);
  const w = Math.abs(px1 - px0);
  const h = Math.abs(py1 - py0);

  emCtx.fillRect(x, y, w, h);
  emCtx.strokeRect(x, y, w, h);
  emCtx.restore();
}

function drawNodeMarkers() {
  if (!latestTelemetry) return;
  emCtx.save();

  function drawMarker(x, y, label, color) {
    emCtx.beginPath();
    emCtx.arc(x, y, 13, 0, 2 * Math.PI);
    emCtx.fillStyle = `${color}88`;
    emCtx.fill();
    emCtx.strokeStyle = color;
    emCtx.lineWidth = 2.0;
    emCtx.stroke();
    emCtx.fillStyle = "#ffffff";
    emCtx.font = "bold 9px sans-serif";
    emCtx.textAlign = "center";
    emCtx.textBaseline = "middle";
    emCtx.fillText(label, x, y);
  }

  if (latestTelemetry.transmitters) {
    latestTelemetry.transmitters.forEach((tx) => {
      drawMarker(toPxX(tx.x), toPxY(tx.y), `TX${tx.id}`, "#ef4444");
    });
  }
  if (latestTelemetry.receivers) {
    latestTelemetry.receivers.forEach((rx) => {
      drawMarker(toPxX(rx.x), toPxY(rx.y), `RX${rx.id}`, "#0ea5e9");
    });
  }
  if (latestTelemetry.observation_points) {
    latestTelemetry.observation_points.forEach((op) => {
      drawMarker(toPxX(op.x), toPxY(op.y), `OBS${op.id}`, "#10b981");
    });
  }
  emCtx.restore();
}

// -----------------------------------------------------------------
// LINK EVALUATORS & INDIVIDUAL TELEMETRY CARDS
// -----------------------------------------------------------------
function renderLinkEvaluators(data) {
  const container = document.getElementById("linkEvaluatorsContainer");
  const block = document.getElementById("linkEvaluatorsBlock");
  const telemetryBlock = document.getElementById("linkTelemetryBlock");
  if (!container || !block) return;

  if (telemetryBlock) telemetryBlock.style.display = "none";

  if (document.activeElement && (document.activeElement.classList.contains("link-tx") || document.activeElement.classList.contains("link-rx"))) {
    return;
  }

  if (!data.link_evaluators || data.link_evaluators.length === 0) {
    block.style.display = "none";
    container.innerHTML = "";
    return;
  }

  block.style.display = "flex";
  container.innerHTML = "";

  data.link_evaluators.forEach((link) => {
    let txOpts = "", rxOpts = "";
    data.transmitters?.forEach(t => {
      txOpts += `<option value="${t.id}" ${t.id === link.tx_id ? 'selected' : ''}>TX ${t.id}</option>`;
    });
    data.receivers?.forEach(r => {
      rxOpts += `<option value="${r.id}" ${r.id === link.rx_id ? 'selected' : ''}>RX ${r.id}</option>`;
    });

    const card = document.createElement("div");
    card.className = "node-card";
    card.style.cssText = "background:#161d2d; border:1.5px solid #f59e0b; border-radius:6px; padding:8px; display:flex; flex-direction:column; gap:6px; margin-bottom:6px;";
    
    card.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; width:100%;">
        <span style="color:#f59e0b; font-size:11.5px; font-weight:700;">Link Evaluator #${link.id}</span>
        <button class="danger" style="padding:2px 8px; font-size:9.5px; cursor:pointer;" onclick="safeSend({type:'remove_link_evaluator', lid:${link.id}})">Del</button>
      </div>
      <div style="display:flex; gap:6px; font-size:10.5px; align-items:center;">
        <label style="flex:1; margin:0;">TX: <select class="preset-select link-tx" data-lid="${link.id}" style="width:100%; margin-top:2px;">${txOpts}</select></label>
        <label style="flex:1; margin:0;">RX: <select class="preset-select link-rx" data-lid="${link.id}" style="width:100%; margin-top:2px;">${rxOpts}</select></label>
      </div>
      <div style="background:#090d15; border-radius:4px; padding:6px; display:grid; grid-template-columns: 1fr 1fr; gap:4px; font-size:10px; margin-top:4px;">
        <div>Sim Time: <span style="color:#38bdf8; font-weight:bold;">${(data.time_ns || 0).toFixed(1)} ns</span></div>
        <div>Bits: <span style="color:#38bdf8; font-weight:bold;">${link.bits_compared || 0}</span></div>
        <div>Errors: <span style="color:#ef4444; font-weight:bold;">${link.bit_errors || 0}</span></div>
        <div>BER: <span style="color:#f8fafc; font-weight:bold;">${Number(link.ber || 0).toFixed(4)}</span></div>
        <div style="grid-column: span 2;">Delay: <span style="color:#10b981; font-weight:bold;">${(link.delay_ns || 0).toFixed(1)} ns</span></div>
      </div>
    `;

    card.querySelector(".link-tx").onchange = (e) => updateLinkPair(link.id, e.target.value, card.querySelector(".link-rx").value);
    card.querySelector(".link-rx").onchange = (e) => updateLinkPair(link.id, card.querySelector(".link-tx").value, e.target.value);
    container.appendChild(card);
  });
}

function updateLinkPair(lid, txId, rxId) {
  safeSend({ type: "update_link_evaluator", lid: parseInt(lid), tx_id: parseInt(txId), rx_id: parseInt(rxId) });
}

// -----------------------------------------------------------------
// CANVAS CLICK & DRAG BOX DRAWING & INSPECTION INTERACTION
// -----------------------------------------------------------------
if (emCanvas) {
  emCanvas.addEventListener("click", (e) => {
    if (!latestTelemetry) return;
    const rect = emCanvas.getBoundingClientRect();
    const cx = toMetersX(((e.clientX - rect.left) / rect.width) * fieldWidth);
    const cy = toMetersY(((e.clientY - rect.top) / rect.height) * fieldHeight);

    // 1. Check if clicking an existing placed material block
    if (latestTelemetry.materials) {
      for (let i = 0; i < latestTelemetry.materials.length; i++) {
        const m = latestTelemetry.materials[i];
        if (cx >= m.x_min && cx <= m.x_max && cy >= m.y_min && cy <= m.y_max) {
          inspectMaterial(i);
          return;
        }
      }
    }

    // 2. Check transmitters
    if (latestTelemetry.transmitters) {
      for (const tx of latestTelemetry.transmitters) {
        if (Math.hypot(cx - tx.x, cy - tx.y) < 0.7) {
          openNodeParameterPopup("tx", tx.id, e.clientX, e.clientY);
          return;
        }
      }
    }
    // 3. Check receivers
    if (latestTelemetry.receivers) {
      for (const rx of latestTelemetry.receivers) {
        if (Math.hypot(cx - rx.x, cy - rx.y) < 0.7) {
          openNodeParameterPopup("rx", rx.id, e.clientX, e.clientY);
          return;
        }
      }
    }
    // 4. Check observation points
    if (latestTelemetry.observation_points) {
      for (const op of latestTelemetry.observation_points) {
        if (Math.hypot(cx - op.x, cy - op.y) < 0.7) {
          openScopeModal("obs", op.id);
          return;
        }
      }
    }
  });

  emCanvas.addEventListener("mousedown", (e) => {
    if (!latestTelemetry) return;
    const rect = emCanvas.getBoundingClientRect();
    const cx = toMetersX(((e.clientX - rect.left) / rect.width) * fieldWidth);
    const cy = toMetersY(((e.clientY - rect.top) / rect.height) * fieldHeight);

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
          isMouseDownOnCanvas = true;
          return;
        }
      }
    }
    if (latestTelemetry.observation_points) {
      for (const op of latestTelemetry.observation_points) {
        if (Math.hypot(cx - op.x, cy - op.y) < 0.6) {
          draggedNode = { type: "obs", id: op.id };
          isMouseDownOnCanvas = true;
          return;
        }
      }
    }

    // Otherwise, start dragging to draw a permanent material box!
    isDrawingBox = true;
    boxStartCoord = { x: cx, y: cy };
    boxCurrentCoord = { x: cx, y: cy };
    isMouseDownOnCanvas = true;
  });

  emCanvas.addEventListener("mousemove", (e) => {
    const rect = emCanvas.getBoundingClientRect();
    const cx = toMetersX(((e.clientX - rect.left) / rect.width) * fieldWidth);
    const cy = toMetersY(((e.clientY - rect.top) / rect.height) * fieldHeight);
    const coordsEl = document.getElementById("canvasCoordsTelemetry");
    if (coordsEl) coordsEl.textContent = `Coords: (${cx.toFixed(2)} m, ${cy.toFixed(2)} m)`;

    if (isMouseDownOnCanvas) {
      if (draggedNode) {
        const d = Math.hypot(e.clientX - startMousePos.x, e.clientY - startMousePos.y);
        if (d > 5) {
          const posX = Math.max(0.5, Math.min(9.5, cx));
          const posY = Math.max(0.5, Math.min(9.5, cy));
          if (draggedNode.type === "tx") safeSend({ type: "move_tx", tx_id: draggedNode.id, x: posX, y: posY });
          else if (draggedNode.type === "rx") safeSend({ type: "move_rx", rx_id: draggedNode.id, x: posX, y: posY });
          else if (draggedNode.type === "obs") safeSend({ type: "move_obs", oid: draggedNode.id, x: posX, y: posY });
        }
      } else if (isDrawingBox) {
        boxCurrentCoord = { x: cx, y: cy };
      }
    }
  });

  window.addEventListener("mouseup", (e) => {
    if (isDrawingBox && isMouseDownOnCanvas) {
      isDrawingBox = false;
      const rect = emCanvas.getBoundingClientRect();
      const endX = toMetersX(((e.clientX - rect.left) / rect.width) * fieldWidth);
      const endY = toMetersY(((e.clientY - rect.top) / rect.height) * fieldHeight);

      const x_min = Math.min(boxStartCoord.x, endX);
      const x_max = Math.max(boxStartCoord.x, endX);
      const y_min = Math.min(boxStartCoord.y, endY);
      const y_max = Math.max(boxStartCoord.y, endY);

      if (Math.abs(x_max - x_min) > 0.05 && Math.abs(y_max - y_min) > 0.05) {
        const matType = document.getElementById("materialSelect")?.value || "concrete";
        let name = matType;
        let perm = 15.0, mu = 1.0, cond = 0.0;

        if (matType === "glass") {
          perm = 4.0;
        } else if (matType === "wood") {
          perm = 2.0;
        } else if (matType === "water") {
          perm = 80.0;
        } else if (matType === "custom") {
          name = customMaterialConfig.name || "custom";
          perm = customMaterialConfig.permittivity;
          mu = customMaterialConfig.permeability;
          cond = customMaterialConfig.conductivity;
        }

        safeSend({
          type: "add_material",
          name: name,
          x_min: x_min,
          x_max: x_max,
          y_min: y_min,
          y_max: y_max,
          relative_permittivity: perm,
          relative_permeability: mu,
          conductivity: cond
        });
      }
    }
    isMouseDownOnCanvas = false;
    draggedNode = null;
  });
}

// Material Property Inspector Function
function inspectMaterial(idx) {
  if (!latestTelemetry || !latestTelemetry.materials || !latestTelemetry.materials[idx]) return;
  const m = latestTelemetry.materials[idx];
  
  let existing = document.getElementById("matInspectModal");
  if (existing) existing.remove();

  const modal = document.createElement("div");
  modal.id = "matInspectModal";
  modal.className = "draggable-modal";
  modal.style.cssText = `top: 150px; left: 150px; width: 320px; display: flex; z-index: ${++activeModalZIndex}; border: 1.5px solid #f59e0b; background: #0d121c; border-radius: 8px;`;

  modal.innerHTML = `
    <div class="modal-header" style="background:#1e293b; padding:8px 12px; display:flex; justify-content:space-between; align-items:center; cursor:grab;">
      <span style="font-size:12px; font-weight:bold; color:#f59e0b;">Material: ${(m.name || "mat").toUpperCase()}</span>
      <button class="icon-btn" onclick="document.getElementById('matInspectModal').remove()">&times;</button>
    </div>
    <div style="padding:14px; display:flex; flex-direction:column; gap:8px; font-size:11px;">
      <div style="display:flex; justify-content:space-between;"><span>Rel. Permittivity (ε_r):</span><strong style="color:#f8fafc;">${m.relative_permittivity || 15}</strong></div>
      <div style="display:flex; justify-content:space-between;"><span>Rel. Permeability (μ_r):</span><strong style="color:#f8fafc;">${m.relative_permeability || 1}</strong></div>
      <div style="display:flex; justify-content:space-between;"><span>Conductivity (σ):</span><strong style="color:#f8fafc;">${m.conductivity || 0} S/m</strong></div>
      <div style="display:flex; justify-content:space-between;"><span>X Range:</span><strong style="color:#38bdf8;">[${m.x_min.toFixed(2)}m - ${m.x_max.toFixed(2)}m]</strong></div>
      <div style="display:flex; justify-content:space-between;"><span>Y Range:</span><strong style="color:#38bdf8;">[${m.y_min.toFixed(2)}m - ${m.y_max.toFixed(2)}m]</strong></div>
      <button class="danger" style="margin-top:8px; padding:6px; font-weight:bold; cursor:pointer;" onclick="safeSend({type:'remove_material', mat_id:${m.id !== undefined ? m.id : idx}}); document.getElementById('matInspectModal').remove();">Delete Material</button>
    </div>
  `;

  document.body.appendChild(modal);
  makeModalDraggable(modal);
}

// -----------------------------------------------------------------
// NEAT CUSTOM MATERIAL MODAL POPUP
// -----------------------------------------------------------------
function openCustomMaterialModal() {
  let existing = document.getElementById("customMatModal");
  if (existing) existing.remove();

  const modal = document.createElement("div");
  modal.id = "customMatModal";
  modal.className = "draggable-modal";
  modal.style.cssText = `top: 150px; left: 150px; width: 360px; display: flex; z-index: ${++activeModalZIndex}; border: 1.5px solid #38bdf8; background: #0d121c; border-radius: 8px;`;

  modal.innerHTML = `
    <div class="modal-header" style="background:#1e293b; padding:8px 12px; display:flex; justify-content:space-between; align-items:center; cursor:grab;">
      <span style="font-size:12px; font-weight:bold; color:#38bdf8;">Configure Custom Material</span>
      <button class="icon-btn" onclick="document.getElementById('customMatModal').remove()">&times;</button>
    </div>
    <div style="padding:14px; display:flex; flex-direction:column; gap:10px; font-size:11px;">
      <label>Material Name:
        <input type="text" id="modalMatName" value="${customMaterialConfig.name}" style="margin-top:2px;">
      </label>
      <label>Relative Permittivity (ε_r):
        <input type="number" id="modalPerm" value="${customMaterialConfig.permittivity}" step="0.5" min="1.0" style="margin-top:2px;">
      </label>
      <label>Relative Permeability (μ_r):
        <input type="number" id="modalMu" value="${customMaterialConfig.permeability}" step="0.1" min="1.0" style="margin-top:2px;">
      </label>
      <label>Conductivity (σ S/m):
        <input type="number" id="modalCond" value="${customMaterialConfig.conductivity}" step="0.01" min="0.0" style="margin-top:2px;">
      </label>
      <button class="success" style="margin-top:8px; padding:6px; font-weight:bold; cursor:pointer;" onclick="saveCustomMaterialConfig()">Save & Apply Brush</button>
    </div>
  `;

  document.body.appendChild(modal);
  makeModalDraggable(modal);
}

function saveCustomMaterialConfig() {
  customMaterialConfig.name = document.getElementById("modalMatName")?.value || "custom";
  customMaterialConfig.permittivity = parseFloat(document.getElementById("modalPerm")?.value) || 10.0;
  customMaterialConfig.permeability = parseFloat(document.getElementById("modalMu")?.value) || 1.0;
  customMaterialConfig.conductivity = parseFloat(document.getElementById("modalCond")?.value) || 0.0;

  const sel = document.getElementById("materialSelect");
  if (sel) sel.value = "custom";

  document.getElementById("customMatModal")?.remove();
}

const btnOpenMatModal = document.getElementById("btnOpenMatModal");
if (btnOpenMatModal) {
  btnOpenMatModal.onclick = (e) => {
    e.preventDefault();
    openCustomMaterialModal();
  };
}

// -----------------------------------------------------------------
// NODE PARAMETER POPUP & SCOPES
// -----------------------------------------------------------------
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
  popup.style.cssText = `top: ${Math.min(window.innerHeight - 380, mouseY)}px; left: ${Math.min(window.innerWidth - 320, mouseX)}px; width: 310px; display: flex; border: 1.5px solid ${type === "tx" ? "#ef4444" : "#38bdf8"}; background: #0d121c; border-radius: 8px;`;

  const title = type === "tx" ? `Transmitter ${id} Parameters` : `Receiver ${id} Parameters`;
  const headerColor = type === "tx" ? "#ef4444" : "#0ea5e9";
  const currentFc = ((nodeData.fc || 1e9) / 1e9).toFixed(2);
  const currentRb = ((nodeData.rb || 500e6) / 1e6).toFixed(0);
  const currentAmp = type === "tx" ? (nodeData.amp || 2.0).toFixed(1) : null;
  const currentWindowSize = 20.0;

  popup.innerHTML = `
    <div class="modal-header" id="paramHeader" style="background:#1e293b; padding:6px 10px; display:flex; justify-content:space-between; align-items:center; cursor:grab;">
      <span style="font-size:11px; font-weight:bold; color:${headerColor};">${title}</span>
      <button class="icon-btn" id="btnClosePopup">&times;</button>
    </div>
    <div style="padding: 10px; display: flex; flex-direction: column; gap: 8px; font-size: 11px;">
      <div style="color:#94a3b8; display:flex; justify-content:space-between;">
        <span>Position (X, Y):</span>
        <span style="color:#f8fafc; font-family:monospace;">(${(nodeData.x || 0).toFixed(2)}m, ${(nodeData.y || 0).toFixed(2)}m)</span>
      </div>
      <label>${type === "tx" ? "Carrier Frequency (fc)" : "Tuned Frequency"}: <span id="valFc" style="color:#f8fafc; font-weight:bold;">${currentFc}</span> GHz
        <input type="range" id="rangeFc" min="0.5" max="3.0" step="0.1" value="${currentFc}" style="width:100%; accent-color:${headerColor};">
      </label>
      <label>Bit Rate (rb): <span id="valRb" style="color:#f8fafc; font-weight:bold;">${currentRb}</span> Mbps
        <input type="range" id="rangeRb" min="50" max="1000" step="50" value="${currentRb}" style="width:100%; accent-color:${headerColor};">
      </label>
      ${type === "tx" ? `<label>Amplitude (amp): <span id="valAmp" style="color:#f8fafc; font-weight:bold;">${currentAmp}</span> V
        <input type="range" id="rangeAmp" min="0.5" max="5.0" step="0.5" value="${currentAmp}" style="width:100\%; accent-color:${headerColor};">
      </label>` : ''}
      
      <label>Window Size / Duration (ns):
        <input type="number" id="inputWindowSize" value="${currentWindowSize}" step="1.0" min="5.0" max="100.0" style="margin-top:2px;">
      </label>

      <div style="display:flex; justify-content:space-between; margin-top:6px; border-top:1px solid #1e293b; padding-top:8px;">
        <button class="danger" id="btnDeleteNode" style="padding:4px 10px; font-size:10px;">Delete Node</button>
        <button class="success" id="btnOpenScope" style="padding:4px 10px; font-size:10px;">Probe / Plots & FFT</button>
      </div>
    </div>
  `;

  document.body.appendChild(popup);
  makeModalDraggable(popup);

  popup.querySelector("#btnClosePopup").onclick = () => popup.remove();
  popup.querySelector("#btnDeleteNode").onclick = () => {
    safeSend({ type: type === "tx" ? "remove_transmitter" : "remove_receiver", [type === 'tx' ? 'tx_id' : 'rx_id']: id });
    popup.remove();
  };
  popup.querySelector("#btnOpenScope").onclick = () => {
    popup.remove();
    openScopeModal(type, id);
  };

  popup.querySelector("#rangeFc").oninput = (e) => {
    popup.querySelector("#valFc").textContent = e.target.value;
    const newFc = parseFloat(e.target.value) * 1e9;
    safeSend({ type: type === "tx" ? "set_tx_params" : "set_rx_params", [type === 'tx' ? 'tx_id' : 'rx_id']: id, fc: newFc });
  };
  popup.querySelector("#rangeRb").oninput = (e) => {
    popup.querySelector("#valRb").textContent = e.target.value;
    const newRb = parseFloat(e.target.value) * 1e6;
    safeSend({ type: type === "tx" ? "set_tx_params" : "set_rx_params", [type === 'tx' ? 'tx_id' : 'rx_id']: id, rb: newRb });
  };
  if (type === "tx") {
    popup.querySelector("#rangeAmp").oninput = (e) => {
      popup.querySelector("#valAmp").textContent = e.target.value;
      safeSend({ type: "set_tx_params", tx_id: id, amp: parseFloat(e.target.value) });
    };
  }
  popup.querySelector("#inputWindowSize").onchange = (e) => {
    const winNs = parseFloat(e.target.value) || 20.0;
    safeSend({ type: type === "tx" ? "set_tx_params" : "set_rx_params", [type === 'tx' ? 'tx_id' : 'rx_id']: id, window_ns: winNs });
  };
}

// -----------------------------------------------------------------
// INDEPENDENT MULTI-SCOPE WINDOW ARCHITECTURE
// -----------------------------------------------------------------
const openScopes = new Map();

function openScopeModal(type, id) {
  const modalKey = `${type}_${id}`;
  if (openScopes.has(modalKey)) {
    openScopes.get(modalKey).modal.style.zIndex = ++activeModalZIndex;
    return;
  }

  const modal = document.createElement("div");
  modal.className = "draggable-modal";
  modal.style.cssText = `top: ${100 + (openScopes.size * 30)}px; left: ${100 + (openScopes.size * 30)}px; width: 680px; display: flex; z-index: ${++activeModalZIndex}; border: 1.5px solid ${type === 'obs' ? '#10b981' : (type === 'tx' ? '#ef4444' : '#38bdf8')}; background: #0d121c; border-radius: 8px;`;

  let stages = [];
  let title = "";
  if (type === "tx") {
    title = `Transmitter ${id} Scope & FFT`;
    stages = [
      { key: "symbols", label: "Bits" },
      { key: "shaped", label: "Baseband" },
      { key: "bpsk", label: "BPSK RF" },
      { key: "spectrum", label: "FFT Spectrum" }
    ];
  } else if (type === "rx") {
    title = `Receiver ${id} Scope & FFT`;
    stages = [
      { key: "rx_raw", label: "Antenna" },
      { key: "rx_bpf", label: "BP Filter" },
      { key: "rx_matched", label: "Matched" },
      { key: "eye_diagram", label: "Eye Diagram" },
      { key: "spectrum", label: "FFT Spectrum" }
    ];
  } else {
    title = `Observation Point ${id} Plots & FFT`;
    stages = [
      { key: "waveform", label: "Waveform" },
      { key: "spectrum", label: "FFT Spectrum" }
    ];
  }

  const scopeData = { modal, ctx: null, type, id, activeStage: stages[0].key };

  let tabsHTML = "";
  stages.forEach((stg, i) => {
    tabsHTML += `<div class="scope-tab ${i === 0 ? 'active' : ''}" data-key="${stg.key}" style="padding:7px 10px; font-size:11px; cursor:pointer; color:#94a3b8; flex:1; text-align:center; font-weight:600;">${stg.label}</div>`;
  });

  let extraControlsHtml = "";
  if (type === "obs") {
    extraControlsHtml = `
      <div style="display:flex; align-items:center; gap:6px; font-size:11px; color:#94a3b8;">
        <span>Window (ns):</span>
        <input type="number" id="obsWinInput_${id}" value="15" min="5" max="100" style="width:60px; padding:3px; font-size:11px; background:#101318; border:1px solid #2c2f37; color:#fff; border-radius:3px;">
      </div>
    `;
  }

  let deleteBtnHtml = `<button class="danger" id="btnDeleteScopeNode" style="padding:5px 12px; font-size:10.5px;">Delete ${type.toUpperCase()}</button>`;

  modal.innerHTML = `
    <div class="modal-header" style="background:#1e293b; padding:8px 12px; display:flex; justify-content:space-between; align-items:center; cursor:grab;">
      <span style="font-size:12px; font-weight:bold; color:${type === 'obs' ? '#10b981' : (type === 'tx' ? '#ef4444' : '#38bdf8')};">${title}</span>
      <button class="icon-btn" style="background:transparent; border:none; color:#fff; cursor:pointer; font-size:15px;">&times;</button>
    </div>
    <div class="scope-tabs" style="display:flex; background:#090d15; border-bottom:1px solid #1f2937;">${tabsHTML}</div>
    <div class="scope-body" style="padding:12px; display:flex; flex-direction:column; gap:10px;">
      <div style="background:#040711; border:1px solid #1f2937; border-radius:4px; padding:6px; position:relative;">
        <canvas width="640" height="240" style="width:100%; height:240px; display:block;"></canvas>
      </div>
      <div style="font-size:11px; color:#64748b; display:flex; justify-content:space-between; align-items:center;">
        ${extraControlsHtml}
        ${deleteBtnHtml}
      </div>
    </div>
  `;

  document.body.appendChild(modal);
  makeModalDraggable(modal);

  const canvas = modal.querySelector("canvas");
  scopeData.ctx = canvas.getContext("2d");

  if (type === "obs") {
    const winInput = modal.querySelector(`#obsWinInput_${id}`);
    if (winInput) {
      winInput.onchange = (e) => {
        const val = parseFloat(e.target.value) || 15.0;
        safeSend({ type: "set_obs_window", oid: id, window_ns: val });
      };
    }
  }

  modal.querySelector("#btnDeleteScopeNode").onclick = () => {
    if (type === "tx") safeSend({ type: "remove_transmitter", tx_id: id });
    else if (type === "rx") safeSend({ type: "remove_receiver", rx_id: id });
    else if (type === "obs") safeSend({ type: "remove_observation_point", oid: id });
    modal.remove();
    openScopes.delete(modalKey);
  };

  modal.querySelector(".icon-btn").onclick = () => {
    modal.remove();
    openScopes.delete(modalKey);
  };

  modal.querySelectorAll(".scope-tab").forEach(tab => {
    tab.onclick = () => {
      modal.querySelectorAll(".scope-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      scopeData.activeStage = tab.getAttribute("data-key");
      renderModalCanvas(scopeData);
    };
  });

  openScopes.set(modalKey, scopeData);
  renderModalCanvas(scopeData);
}

function renderModalCanvas(scopeData) {
  if (!latestTelemetry) return;
  const ctx = scopeData.ctx;
  const w = ctx.canvas.width; const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);

  let values = [];
  if (scopeData.type === "tx") {
    const tx = latestTelemetry.transmitters?.find(t => t.id === scopeData.id);
    if (!tx) return;
    if (scopeData.activeStage === "spectrum") { drawAdvancedPlot(ctx, tx.spectrum, w, h, "Frequency (GHz)", "Amplitude", true); return; }
    values = tx[scopeData.activeStage] || tx.bpsk || [];
  } else if (scopeData.type === "rx") {
    const rx = latestTelemetry.receivers?.find(r => r.id === scopeData.id);
    if (!rx) return;
    if (scopeData.activeStage === "spectrum") { drawAdvancedPlot(ctx, rx.spectrum || latestTelemetry.spectrum, w, h, "Frequency (GHz)", "Amplitude", true); return; }
    if (scopeData.activeStage === "eye_diagram") { drawEyeDiagramWithAxes(ctx, rx.rx_matched || latestTelemetry.rx_matched || [], w, h); return; }
    values = rx[scopeData.activeStage] || rx.rx_raw || [];
  } else if (scopeData.type === "obs") {
    const op = latestTelemetry.observation_points?.find(o => o.id === scopeData.id);
    if (!op) return;
    if (scopeData.activeStage === "spectrum") { drawAdvancedPlot(ctx, op.spectrum, w, h, "Frequency (GHz)", "Amplitude", true); return; }
    values = op.waveform || op.buffered_values || op.samples || [];
  }
  drawAdvancedPlot(ctx, values, w, h, "Sample Index", "Amplitude (V)", false);
}

function drawAdvancedPlot(ctx, values, w, h, xLabel, yLabel, isSpectrum) {
  const padLeft = 55, padBottom = 45, padTop = 15, padRight = 20;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  ctx.strokeStyle = "#162032"; ctx.lineWidth = 1;
  ctx.fillStyle = "#94a3b8"; ctx.font = "10px monospace";

  let maxVal = 1.0, minVal = isSpectrum ? 0.0 : -1.0;
  if (values && values.length > 0) {
    const peak = Math.max(...values.map(Math.abs), 0.01);
    maxVal = isSpectrum ? peak * 1.1 : peak * 1.25;
    minVal = isSpectrum ? 0.0 : -maxVal;
  }

  for (let i = 0; i <= 4; i++) {
    const gy = padTop + (plotH / 4) * i;
    const val = maxVal - (i / 4) * (maxVal - minVal);
    ctx.beginPath(); ctx.moveTo(padLeft, gy); ctx.lineTo(w - padRight, gy); ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(val.toFixed(2), padLeft - 8, gy);
  }

  for (let i = 0; i <= 4; i++) {
    const gx = padLeft + (plotW / 4) * i;
    ctx.beginPath(); ctx.moveTo(gx, padTop); ctx.lineTo(gx, h - padBottom); ctx.stroke();
    if (values && values.length > 0) {
      const xVal = ((i / 4) * values.length).toFixed(0);
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(xVal, gx, h - padBottom + 6);
    }
  }

  ctx.save();
  ctx.fillStyle = "#cbd5e1"; ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, padLeft + plotW / 2, h - 14);
  ctx.translate(14, padTop + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  if (!values || values.length < 2) {
    ctx.fillStyle = "#64748b"; ctx.font = "11.5px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Waiting for telemetry data...", padLeft + plotW / 2, padTop + plotH / 2);
    return;
  }

  if (isSpectrum) {
    ctx.fillStyle = "#facc15";
    const barW = plotW / values.length;
    values.forEach((v, i) => {
      const normH = Math.max(0, Math.min(1, v / (maxVal || 1.0)));
      const barH = normH * plotH;
      ctx.fillRect(padLeft + (i * barW), (padTop + plotH) - barH, Math.max(1, barW - 1), barH);
    });
  } else {
    ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 1.8; ctx.beginPath();
    values.forEach((v, i) => {
      const x = padLeft + (i / (values.length - 1)) * plotW;
      const normY = Math.max(0, Math.min(1, (v - minVal) / (maxVal - minVal)));
      const y = (padTop + plotH) - (normY * plotH);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

function drawEyeDiagramWithAxes(ctx, values, w, h) {
  const padLeft = 55, padBottom = 45, padTop = 15, padRight = 20;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  ctx.strokeStyle = "#162032"; ctx.lineWidth = 1;
  ctx.fillStyle = "#94a3b8"; ctx.font = "10px monospace";

  for (let i = 0; i <= 4; i++) {
    const gy = padTop + (plotH / 4) * i;
    const val = 1.8 - (i / 4) * 3.6;
    ctx.beginPath(); ctx.moveTo(padLeft, gy); ctx.lineTo(w - padRight, gy); ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(val.toFixed(1), padLeft - 8, gy);
  }

  for (let i = 0; i <= 4; i++) {
    const gx = padLeft + (plotW / 4) * i;
    ctx.beginPath(); ctx.moveTo(gx, padTop); ctx.lineTo(gx, h - padBottom); ctx.stroke();
  }

  ctx.save();
  ctx.fillStyle = "#cbd5e1"; ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Time (2x Symbol Period)", padLeft + plotW / 2, h - 14);
  ctx.translate(14, padTop + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Amplitude (V)", 0, 0);
  ctx.restore();

  if (!values || values.length < 32) {
    ctx.fillStyle = "#64748b"; ctx.font = "11.5px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Accumulating eye diagram samples...", padLeft + plotW / 2, padTop + plotH / 2);
    return;
  }

  const symLen = 16;
  ctx.strokeStyle = "rgba(232, 121, 249, 0.5)";
  ctx.lineWidth = 1.5;
  for (let s = 0; s < values.length - symLen * 2; s += symLen) {
    ctx.beginPath();
    for (let i = 0; i < symLen * 2; i++) {
      const x = padLeft + (i / (symLen * 2 - 1)) * plotW;
      const normY = Math.max(0, Math.min(1, (values[s + i] + 1.8) / 3.6));
      const y = (padTop + plotH) - (normY * plotH);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

function makeModalDraggable(modal) {
  const header = modal.querySelector(".modal-header");
  let isDrag = false, startX = 0, startY = 0;
  header.onmousedown = (e) => {
    isDrag = true; startX = e.clientX - modal.offsetLeft; startY = e.clientY - modal.offsetTop;
    modal.style.zIndex = ++activeModalZIndex;
  };
  window.addEventListener("mouseup", () => isDrag = false);
  window.addEventListener("mousemove", (e) => {
    if (!isDrag) return;
    modal.style.left = `${e.clientX - startX}px`;
    modal.style.top = `${e.clientY - startY}px`;
  });
}

function refreshOpenModals() {
  openScopes.forEach((scopeData) => {
    renderModalCanvas(scopeData);
  });
}

// -----------------------------------------------------------------
// UI ENGINE & TOOLBAR CONTROLS
// -----------------------------------------------------------------
const btnStart = document.getElementById("btnStart");
if (btnStart) btnStart.onclick = () => safeSend({ type: "start" });

const btnPause = document.getElementById("btnPause");
if (btnPause) btnPause.onclick = () => safeSend({ type: "pause" });

const btnReset = document.getElementById("btnReset");
if (btnReset) btnReset.onclick = () => safeSend({ type: "reset" });

const btnClearWalls = document.getElementById("btnClearWalls");
if (btnClearWalls) btnClearWalls.onclick = () => safeSend({ type: "clear_walls" });

const btnClearWallsAlt = document.getElementById("btnClearWallsAlt");
if (btnClearWallsAlt) btnClearWallsAlt.onclick = () => safeSend({ type: "clear_walls" });

const applyConfigBtn = document.getElementById("applyConfigBtn");
if (applyConfigBtn) {
  applyConfigBtn.onclick = () => {
    const res = Math.max(100, parseInt(document.getElementById("resolutionInput")?.value) || 1000);
    const dtMult = parseFloat(document.getElementById("dtMultiplierInput")?.value) || 0.25;
    safeSend({ type: "set_resolution", width: res, height: res, dt_multiplier: dtMult });
  };
}

const btnAddTx = document.getElementById("btnAddTransmitterBtn");
if (btnAddTx) {
  btnAddTx.onclick = () => {
    const count = latestTelemetry && latestTelemetry.transmitters ? latestTelemetry.transmitters.length : 0;
    const staggeredX = 1.5 + (count * 1.5) % 7.0;
    const staggeredY = 7.0 - Math.floor(count / 4) * 1.5;
    safeSend({ type: "add_transmitter", x: staggeredX, y: staggeredY });
  };
}

const btnAddRx = document.getElementById("btnAddReceiverBtn");
if (btnAddRx) {
  btnAddRx.onclick = () => {
    const count = latestTelemetry && latestTelemetry.receivers ? latestTelemetry.receivers.length : 0;
    const staggeredX = 8.0 - (count * 1.5) % 4.0;
    const staggeredY = 3.0 + (count * 1.2) % 5.0;
    safeSend({ type: "add_receiver", x: staggeredX, y: staggeredY });
  };
}

const btnAddObs = document.getElementById("btnAddObsPointBtn");
if (btnAddObs) {
  btnAddObs.onclick = () => {
    const count = latestTelemetry && latestTelemetry.observation_points ? latestTelemetry.observation_points.length : 0;
    const staggeredX = 4.0 + (count * 1.2) % 3.0;
    const staggeredY = 4.0 + (count * 1.2) % 3.0;
    safeSend({ type: "add_observation_point", x: staggeredX, y: staggeredY, label: `Grid Probe` });
  };
}

const btnAddLinkEval = document.getElementById("btnAddLinkEvalBtn");
if (btnAddLinkEval) {
  btnAddLinkEval.onclick = () => {
    safeSend({ type: "add_link_evaluator" });
  };
}

const scenarioSelect = document.getElementById("scenarioSelect");
if (scenarioSelect) {
  scenarioSelect.onchange = (e) => {
    const val = e.target.value;
    if (val === "los") {
      safeSend({ type: "clear_walls" });
    } else if (val === "concrete_wall") {
      safeSend({ type: "add_material", name: "concrete", x_min: 4.8, x_max: 5.2, y_min: 2.0, y_max: 8.0 });
    }
  };
}