"""
server.py - Full-Stack FDTD Backend
Supports explicit TX-RX link pairing, independent link evaluations, and material rotation masks.
"""

import asyncio
import os
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.simulation_space import SimulationSpace
from src.wave_solver import WaveSolver
from src.materials import Material
from src.transmitter import Transmitter
from src.receiver import Receiver
from src.link_evaluator import LinkEvaluator
from src.observation_point import ObservationPoint


def safe_call(obj, *names, default=0.0):
    for name in names:
        if hasattr(obj, name):
            attr = getattr(obj, name)
            if callable(attr):
                try:
                    return attr()
                except Exception:
                    pass
            else:
                return attr
    return default


class SimulationRuntime:
    def __init__(self):
        self.resolution_x = 600
        self.resolution_y = 600
        self.width = 10.0
        self.height = 10.0
        self.steps_per_frame = 3
        self.target_fps = 30
        self.running = False
        self.view_mode = "field"

        self.transmitters = {}
        self.receivers = {}
        self.evaluators = {}
        self.rx_tx_mapping = {}  # Maps rx_id -> tx_id
        self.materials_list = []
        self.active_rx_id = 0

        self._build()

    def _build(self):
        c = 3.0e8
        dx = self.width / (self.resolution_x - 1)
        dy = self.height / (self.resolution_y - 1)
        dt = 0.85 / (c * np.sqrt((1.0 / dx**2) + (1.0 / dy**2)))

        self.space = SimulationSpace(
            width=self.width,
            height=self.height,
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            dt_stability_multiplier=0.85,
        )
        self.space.set_global_permittivity(8.8541878128e-12)
        self.space.set_global_conductivity(0.0)

        self.transmitters.clear()
        self.receivers.clear()
        self.evaluators.clear()
        self.rx_tx_mapping.clear()

        # Initial default transmitters
        self.add_transmitter(tx_id=0, x=1.5, y=6.5, fc=1.0e9, rb=500.0e6, amp=2.0)
        self.add_transmitter(tx_id=1, x=1.5, y=3.5, fc=1.2e9, rb=250.0e6, amp=2.0)

        # Initial default receiver
        self.add_receiver(rx_id=0, x=8.5, y=5.0, bit_rate=500.0e6)

        self.obs_point = ObservationPoint(
            self.space,
            x=8.5,
            y=5.0,
            buffer_duration=15e-9,
            label="RX Antenna Probe",
        )

        self.wave_solver = WaveSolver(self.space, noise_level=0.0)
        self.space.set_running(True)

    def add_transmitter(self, tx_id, x, y, fc=1.0e9, rb=500.0e6, amp=2.0):
        tx = Transmitter(
            self.space,
            x=float(x),
            y=float(y),
            carrier_frequency=float(fc),
            carrier_amplitude=float(amp),
            bit_rate=float(rb),
            window_duration=20e-9,
        )
        self.transmitters[tx_id] = tx
        self._rebuild_evaluators()

    def add_receiver(self, rx_id, x, y, bit_rate=500.0e6, target_tx_id=0):
        rx = Receiver(
            self.space,
            x=float(x),
            y=float(y),
            tuned_frequency=1.0e9,
            bit_rate=bit_rate,
            observation_window=20e-9,
        )
        self.receivers[rx_id] = rx
        self.rx_tx_mapping[rx_id] = target_tx_id if target_tx_id in self.transmitters else 0
        self._rebuild_evaluators()

    def set_rx_link(self, rx_id, tx_id):
        if rx_id in self.receivers and tx_id in self.transmitters:
            self.rx_tx_mapping[rx_id] = tx_id
            self._rebuild_evaluators()

    def _rebuild_evaluators(self):
        self.evaluators.clear()
        for rid, rx in self.receivers.items():
            target_tx_id = self.rx_tx_mapping.get(rid, 0)
            tx = self.transmitters.get(target_tx_id) or next(iter(self.transmitters.values()), None)
            if not tx:
                continue
            ev = LinkEvaluator(
                transmitter=tx,
                receiver=rx,
                speed_of_light=3.0e8,
                filter_group_delay_samples=8,
                warmup_bits=2,
            )
            ev.sync_receiver_delay()
            self.evaluators[rid] = ev

    def add_material(self, name, x_min, x_max, y_min, y_max, angle=0.0, rel_perm=None, cond=0.0):
        mat = Material(
            simulation_space=self.space,
            name=name,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
        )
        if rel_perm is not None and hasattr(mat, "set_relative_permittivity"):
            mat.set_relative_permittivity(float(rel_perm))
        if cond is not None and hasattr(mat, "set_conductivity"):
            mat.set_conductivity(float(cond))

        mat.apply()

        # Rotation mask support
        angle_rad = np.radians(float(angle))
        if angle_rad != 0.0 and hasattr(self.space, "grid_x") and hasattr(self.space, "grid_y"):
            try:
                cx = (float(x_min) + float(x_max)) / 2.0
                cy = (float(y_min) + float(y_max)) / 2.0
                hx = (float(x_max) - float(x_min)) / 2.0
                hy = (float(y_max) - float(y_min)) / 2.0

                X = self.space.grid_x
                Y = self.space.grid_y

                cos_a = np.cos(-angle_rad)
                sin_a = np.sin(-angle_rad)
                
                dx = X - cx
                dy = Y - cy
                rx = cos_a * dx - sin_a * dy
                ry = sin_a * dx + cos_a * dy

                mask = (np.abs(rx) <= hx) & (np.abs(ry) <= hy)
                perm_val = float(rel_perm) if rel_perm is not None else (15.0 if name == "concrete" else (4.0 if name == "glass" else (2.5 if name == "wood" else 80.0)))
                if hasattr(self.space, "epsilon"):
                    eps0 = 8.8541878128e-12
                    self.space.epsilon[mask] = eps0 * perm_val
            except Exception as rot_err:
                print(f"Rotational mask overlay note: {rot_err}")

        if hasattr(self.wave_solver, "refresh"):
            self.wave_solver.refresh()

        self.materials_list.append({
            "name": name,
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
            "angle": float(angle),
            "speed": safe_call(mat, "get_wave_speed", default=3.0e8),
            "attenuation": safe_call(mat, "get_attenuation", default=0.0),
        })

    def clear_materials(self):
        self.space.clear()
        if hasattr(self.space, "set_global_permittivity"):
            self.space.set_global_permittivity(8.8541878128e-12)
        if hasattr(self.space, "set_global_conductivity"):
            self.space.set_global_conductivity(0.0)
        if hasattr(self.wave_solver, "refresh"):
            self.wave_solver.refresh()
        self.materials_list.clear()

    def step(self):
        for tx in self.transmitters.values():
            tx.transmit()
        self.wave_solver.solve()
        for rx in self.receivers.values():
            rx.receive()
        for ev in self.evaluators.values():
            ev.evaluate()
        self.obs_point.sample()
        self.space.advance_time()

    def field_bytes(self):
        field = self.space.get_current_field()
        if self.view_mode == "energy":
            display_mat = np.sqrt(np.abs(field))
            quantized = np.clip(display_mat * (255.0 / 2.0), 0, 255)
        else:
            quantized = np.clip((field + 2.0) * (255.0 / 4.0), 0, 255)

        field_display = np.flipud(quantized.T)
        return field_display.astype(np.uint8).tobytes()

    def status(self):
        active_rx = self.receivers.get(self.active_rx_id) or next(iter(self.receivers.values()))
        active_ev = self.evaluators.get(self.active_rx_id) or next(iter(self.evaluators.values()), None)

        tx_list = []
        for tid, tx in self.transmitters.items():
            freqs, amps = tx.compute_bpsk_fft()
            fft_spec = amps[freqs <= 3.0e9].tolist() if len(amps) > 0 else []
            tx_list.append({
                "id": tid,
                "x": tx.x,
                "y": tx.y,
                "fc": safe_call(tx, "carrier_frequency", "get_carrier_frequency", default=1.0e9),
                "rb": safe_call(tx, "bit_rate", "get_bit_rate", default=500.0e6),
                "amp": safe_call(tx, "carrier_amplitude", "get_carrier_amplitude", default=2.0),
                "symbols": list(tx.get_bit_values()) if hasattr(tx, "get_bit_values") else [],
                "shaped": list(tx.get_shaped_values()) if hasattr(tx, "get_shaped_values") else [],
                "bpsk": list(tx.get_bpsk_values()) if hasattr(tx, "get_bpsk_values") else [],
                "spectrum": fft_spec,
            })

        rx_list = []
        for rid, r in self.receivers.items():
            ev = self.evaluators.get(rid)
            ber_val = ev.get_bit_error_rate() if ev and ev.get_bit_error_rate() is not None else 0.0
            delay_val = ev.get_estimated_total_delay_seconds() * 1e9 if ev else 0.0
            compared_val = ev.get_total_bits_compared() if ev else 0
            errors_val = ev.get_bit_errors() if ev else 0
            rx_list.append({
                "id": rid,
                "x": r.x,
                "y": r.y,
                "fc": safe_call(r, "tuned_frequency", "get_tuned_frequency", default=1.0e9),
                "rb": safe_call(r, "bit_rate", "get_bit_rate", default=500.0e6),
                "linked_tx": self.rx_tx_mapping.get(rid, 0),
                "ber": ber_val,
                "delay_ns": delay_val,
                "bits_compared": compared_val,
                "bit_errors": errors_val,
            })

        freqs, amps = self.obs_point.compute_fft(window_type="hann")
        fft_data = amps[freqs <= 3.0e9].tolist() if len(amps) > 0 else []
        peak_f, peak_a = self.obs_point.get_peak_frequency(window_type="hann")
        window_energy = safe_call(self.obs_point, "get_windowed_energy", default=0.0)

        active_ber = active_ev.get_bit_error_rate() if active_ev else 0.0

        return {
            "type": "telemetry",
            "running": self.running,
            "time_ns": self.space.time * 1e9,
            "active_rx_id": self.active_rx_id,
            "bits_compared": active_ev.get_total_bits_compared() if active_ev else 0,
            "bit_errors": active_ev.get_bit_errors() if active_ev else 0,
            "ber": active_ber if active_ber is not None else 0.0,
            "est_delay_ns": active_ev.get_estimated_total_delay_seconds() * 1e9 if active_ev else 0.0,
            "transmitters": tx_list,
            "receivers": rx_list,
            "materials": self.materials_list,
            "obs_probe": {
                "peak_freq_ghz": float(peak_f / 1e9),
                "peak_mag": float(peak_a),
                "energy_joules": float(window_energy),
            },
            "rx_raw": list(active_rx.get_received_values()),
            "rx_bpf": list(active_rx.get_filtered_values()),
            "rx_mixed": list(active_rx.get_mixed_values()),
            "rx_matched": list(active_rx.get_baseband_values()),
            "spectrum": fft_data,
        }


runtime = SimulationRuntime()
app = FastAPI()

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return FileResponse("index.html")


def handle_control_message(message: dict):
    try:
        msg_type = message.get("type")

        if msg_type in ("start", "resume"):
            runtime.running = True
        elif msg_type == "pause":
            runtime.running = False
        elif msg_type == "reset":
            runtime.running = False
            runtime._build()
        elif msg_type == "set_view_mode":
            runtime.view_mode = message.get("mode", "field")
        elif msg_type == "add_transmitter":
            new_id = max(runtime.transmitters.keys()) + 1 if runtime.transmitters else 0
            runtime.add_transmitter(new_id, float(message.get("x", 2.0)), float(message.get("y", 5.0)))
        elif msg_type == "add_receiver":
            new_id = max(runtime.receivers.keys()) + 1 if runtime.receivers else 0
            runtime.add_receiver(new_id, float(message.get("x", 8.0)), float(message.get("y", 5.0)), target_tx_id=0)
            runtime.active_rx_id = new_id
        elif msg_type == "set_rx_link":
            runtime.set_rx_link(int(message.get("rx_id", 0)), int(message.get("tx_id", 0)))
        elif msg_type == "select_receiver":
            rx_id = int(message.get("rx_id", 0))
            if rx_id in runtime.receivers:
                runtime.active_rx_id = rx_id
                runtime.obs_point.x = runtime.receivers[rx_id].x
                runtime.obs_point.y = runtime.receivers[rx_id].y
        elif msg_type == "move_tx":
            tx_id = int(message.get("tx_id", 0))
            if tx_id in runtime.transmitters:
                runtime.transmitters[tx_id].set_position(float(message["x"]), float(message["y"]))
                runtime._rebuild_evaluators()
        elif msg_type == "move_rx":
            rx_id = int(message.get("rx_id", runtime.active_rx_id))
            if rx_id in runtime.receivers:
                runtime.receivers[rx_id].set_position(float(message["x"]), float(message["y"]))
                if rx_id in runtime.evaluators:
                    runtime.evaluators[rx_id].sync_receiver_delay()
                if rx_id == runtime.active_rx_id:
                    runtime.obs_point.x = float(message["x"])
                    runtime.obs_point.y = float(message["y"])
        elif msg_type == "set_tx_params":
            tx_id = int(message.get("tx_id", 0))
            if tx_id in runtime.transmitters:
                tx = runtime.transmitters[tx_id]
                if "fc" in message:
                    val = float(message["fc"])
                    if hasattr(tx, "set_carrier_frequency"):
                        tx.set_carrier_frequency(val)
                    else:
                        tx.carrier_frequency = val
                if "rb" in message:
                    val = float(message["rb"])
                    if hasattr(tx, "set_bit_rate"):
                        tx.set_bit_rate(val)
                    else:
                        tx.bit_rate = val
                if "amp" in message:
                    tx.carrier_amplitude = float(message["amp"])
                runtime._rebuild_evaluators()
        elif msg_type == "set_rx_params":
            rx_id = int(message.get("rx_id", runtime.active_rx_id))
            if rx_id in runtime.receivers:
                rx = runtime.receivers[rx_id]
                if "fc" in message:
                    val = float(message["fc"])
                    if hasattr(rx, "set_tuned_frequency"):
                        rx.set_tuned_frequency(val)
                    else:
                        rx.tuned_frequency = val
                if "rb" in message:
                    val = float(message["rb"])
                    if hasattr(rx, "set_bit_rate"):
                        rx.set_bit_rate(val)
                    else:
                        rx.bit_rate = val
                if rx_id in runtime.evaluators:
                    runtime.evaluators[rx_id].sync_receiver_delay()
        elif msg_type == "add_material":
            runtime.add_material(
                name=message.get("name", "concrete"),
                x_min=float(message.get("x_min", 4.0)),
                x_max=float(message.get("x_max", 5.0)),
                y_min=float(message.get("y_min", 2.0)),
                y_max=float(message.get("y_max", 8.0)),
                angle=float(message.get("angle", 0.0)),
                rel_perm=message.get("relative_permittivity"),
                cond=message.get("conductivity", 0.0),
            )
        elif msg_type in ("clear_walls", "clear_materials"):
            runtime.clear_materials()
    except Exception as e:
        print(f"Error handling control message {message}: {e}")


@app.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_json(runtime.status())
    except Exception:
        return

    async def receive_loop():
        try:
            while True:
                msg = await websocket.receive_json()
                handle_control_message(msg)
        except (WebSocketDisconnect, Exception):
            pass

    receive_task = asyncio.create_task(receive_loop())

    try:
        frame_interval = 1.0 / runtime.target_fps
        while True:
            loop_start = time.perf_counter()
            if runtime.running:
                for _ in range(runtime.steps_per_frame):
                    runtime.step()
            
            try:
                await websocket.send_bytes(runtime.field_bytes())
                await websocket.send_json(runtime.status())
            except Exception:
                break  # Exit cleanly if a frame fails to send

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_interval - elapsed))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receive_task.cancel()
