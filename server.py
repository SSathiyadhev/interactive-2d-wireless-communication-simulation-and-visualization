"""
server.py - Full-Stack FDTD Backend with Carrier Wave & Receiver Bits Telemetry Support
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
        self.resolution_x = 1000
        self.resolution_y = 1000
        self.width = 10.0
        self.height = 10.0
        self.dt_multiplier = 0.25
        self.steps_per_frame = 3
        self.target_fps = 30
        self.running = False
        self.view_mode = "field"

        self.transmitters = {}
        self.receivers = {}
        self.observation_points = {}
        self.link_evaluators_dict = {}
        
        self.next_obs_id = 0
        self.next_link_eval_id = 0

        self.materials_list = []
        self._build()

    def _build(self, res_x=None, res_y=None, dt_multiplier=None):
        if res_x is not None:
            self.resolution_x = max(100, int(res_x))
        if res_y is not None:
            self.resolution_y = max(100, int(res_y))
        if dt_multiplier is not None:
            self.dt_multiplier = float(dt_multiplier)

        self.space = SimulationSpace(
            width=self.width,
            height=self.height,
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            dt_stability_multiplier=self.dt_multiplier,
        )
        self.space.set_global_permittivity(8.8541878128e-12)
        self.space.set_global_conductivity(0.0)

        self.transmitters.clear()
        self.receivers.clear()
        self.observation_points.clear()
        self.link_evaluators_dict.clear()

        self.add_transmitter(0, 2.0, 7.0, fc=1.0e9, rb=500.0e6, amp=2.0)
        self.add_receiver(0, 8.0, 5.0, bit_rate=500.0e6)
        self.add_observation_point(5.0, 5.0, label="Grid Probe 0")

        self.wave_solver = WaveSolver(self.space, noise_level=0.0)
        self.space.set_running(True)

    def add_transmitter(self, tx_id, x, y, fc=1.0e9, rb=500.0e6, amp=2.0, window_duration=20e-9):
        tx = Transmitter(
            self.space, x=float(x), y=float(y),
            carrier_frequency=float(fc), carrier_amplitude=float(amp),
            bit_rate=float(rb), window_duration=float(window_duration),
        )
        self.transmitters[tx_id] = tx

    def remove_transmitter(self, tx_id):
        if tx_id in self.transmitters:
            del self.transmitters[tx_id]
            to_del = [lid for lid, data in self.link_evaluators_dict.items() if data["tx_id"] == tx_id]
            for lid in to_del:
                del self.link_evaluators_dict[lid]

    def add_receiver(self, rx_id, x, y, bit_rate=500.0e6, observation_window=20e-9):
        rx = Receiver(
            self.space, x=float(x), y=float(y),
            tuned_frequency=1.0e9, bit_rate=float(bit_rate), observation_window=float(observation_window),
        )
        self.receivers[rx_id] = rx

    def remove_receiver(self, rx_id):
        if rx_id in self.receivers:
            del self.receivers[rx_id]
            to_del = [lid for lid, data in self.link_evaluators_dict.items() if data["rx_id"] == rx_id]
            for lid in to_del:
                del self.link_evaluators_dict[lid]

    def add_observation_point(self, x, y, label=""):
        oid = self.next_obs_id
        self.next_obs_id += 1
        op = ObservationPoint(
            self.space, x=float(x), y=float(y),
            buffer_duration=15e-9, label=label or f"Obs Point {oid}"
        )
        self.observation_points[oid] = op
        return oid

    def remove_observation_point(self, oid):
        if oid in self.observation_points:
            del self.observation_points[oid]

    def add_link_evaluator(self, tx_id, rx_id):
        lid = self.next_link_eval_id
        self.next_link_eval_id += 1
        
        tx = self.transmitters.get(tx_id)
        rx = self.receivers.get(rx_id)
        
        ev = None
        if tx and rx:
            ev = LinkEvaluator(
                transmitter=tx, receiver=rx, speed_of_light=3.0e8,
                filter_group_delay_samples=8, warmup_bits=2,
            )
            ev.sync_receiver_delay()

        self.link_evaluators_dict[lid] = {
            "tx_id": tx_id,
            "rx_id": rx_id,
            "evaluator": ev
        }
        return lid

    def remove_link_evaluator(self, lid):
        if lid in self.link_evaluators_dict:
            del self.link_evaluators_dict[lid]

    def update_link_evaluator_pairing(self, lid, tx_id, rx_id):
        if lid in self.link_evaluators_dict:
            tx = self.transmitters.get(tx_id)
            rx = self.receivers.get(rx_id)
            ev = None
            if tx and rx:
                ev = LinkEvaluator(
                    transmitter=tx, receiver=rx, speed_of_light=3.0e8,
                    filter_group_delay_samples=8, warmup_bits=2,
                )
                ev.sync_receiver_delay()
            self.link_evaluators_dict[lid] = {
                "tx_id": tx_id,
                "rx_id": rx_id,
                "evaluator": ev
            }

    def add_material(self, name, x_min=None, x_max=None, y_min=None, y_max=None, angle=0.0, rel_perm=None, rel_mu=None, cond=0.0):
        if x_min is None or x_max is None or y_min is None or y_max is None:
            idx = len(self.materials_list)
            w_box, h_box = 0.4, 4.0
            x_min = 4.5 + (idx * 1.2) % 4.0
            x_max = x_min + w_box
            y_min = 2.0
            y_max = y_min + h_box

        mat_name = name if name in ["concrete", "glass", "wood", "water"] else "concrete"

        mat = Material(
            simulation_space=self.space, name=mat_name,
            x_min=float(x_min), x_max=float(x_max),
            y_min=float(y_min), y_max=float(y_max),
        )
        if rel_perm is not None and hasattr(mat, "set_relative_permittivity"):
            mat.set_relative_permittivity(float(rel_perm))
        if rel_mu is not None and hasattr(mat, "set_relative_permeability"):
            mat.set_relative_permeability(float(rel_mu))
        if cond is not None and hasattr(mat, "set_conductivity"):
            mat.set_conductivity(float(cond))
        mat.apply()

        if hasattr(self.wave_solver, "refresh"):
            self.wave_solver.refresh()

        mat_id = len(self.materials_list)
        self.materials_list.append({
            "id": mat_id,
            "name": name, 
            "x_min": float(x_min), "x_max": float(x_max),
            "y_min": float(y_min), "y_max": float(y_max), 
            "angle": float(angle),
            "relative_permittivity": float(rel_perm) if rel_perm is not None else 15.0,
            "relative_permeability": float(rel_mu) if rel_mu is not None else 1.0,
            "conductivity": float(cond) if cond is not None else 0.0
        })

    def remove_material(self, mat_id):
        self.materials_list = [
            m for m in self.materials_list
            if m.get("id") != mat_id
        ]
        old_mats = list(self.materials_list)

        self.space.set_global_permittivity(8.8541878128e-12)
        self.space.set_global_permeability(4.0e-7 * np.pi)
        self.space.set_global_conductivity(0.0)
        self.materials_list.clear()

        for m in old_mats:
            self.add_material(
                name=m["name"],
                x_min=m["x_min"],
                x_max=m["x_max"],
                y_min=m["y_min"],
                y_max=m["y_max"],
                angle=m["angle"],
                rel_perm=m["relative_permittivity"],
                rel_mu=m["relative_permeability"],
                cond=m["conductivity"]
            )

        if hasattr(self.wave_solver, "refresh"):
            self.wave_solver.refresh()

    def clear_materials(self):
        self.space.set_global_permittivity(8.8541878128e-12)
        self.space.set_global_permeability(4.0e-7 * np.pi)
        self.space.set_global_conductivity(0.0)
        self.materials_list.clear()

        if hasattr(self.wave_solver, "refresh"):
            self.wave_solver.refresh()

    def step(self):
        for tx in self.transmitters.values():
            tx.transmit()
        self.wave_solver.solve()
        for rx in self.receivers.values():
            rx.receive()
        for ldata in self.link_evaluators_dict.values():
            if ldata["evaluator"]:
                ldata["evaluator"].evaluate()
        for op in self.observation_points.values():
            op.sample()
        self.space.advance_time()

    def field_bytes(self):
        field = self.space.get_current_field()
        if field.shape != (self.resolution_x, self.resolution_y):
            field = field.reshape((self.resolution_x, self.resolution_y))

        if self.view_mode == "energy":
            display_mat = np.sqrt(np.abs(field))
            quantized = np.clip(display_mat * (255.0 / 2.0), 0, 255)
        else:
            quantized = np.clip((field + 2.0) * (255.0 / 4.0), 0, 255)

        return quantized.T.astype(np.uint8).tobytes()

    def status(self):
        tx_list = []
        for tid, tx in self.transmitters.items():
            freqs, amps = tx.compute_bpsk_fft()
            fft_spec = amps[freqs <= 3.0e9].tolist() if len(amps) > 0 else []
            tx_list.append({
                "id": tid, "x": float(tx.x), "y": float(tx.y),
                "fc": safe_call(tx, "carrier_frequency", "fc", default=1.0e9),
                "rb": safe_call(tx, "bit_rate", "rb", default=500.0e6),
                "amp": safe_call(tx, "carrier_amplitude", "amplitude", "amp", default=2.0),
                "symbols": list(tx.get_bit_values()) if hasattr(tx, "get_bit_values") else [],
                "shaped": list(tx.get_shaped_values()) if hasattr(tx, "get_shaped_values") else [],
                "carrier": list(tx.get_carrier_values()) if hasattr(tx, "get_carrier_values") else [],
                "bpsk": list(tx.get_bpsk_values()) if hasattr(tx, "get_bpsk_values") else [],
                "spectrum": fft_spec,
            })

        rx_list = []
        for rid, r in self.receivers.items():
            rx_fft_spec = []
            try:
                raw_vals = np.array(r.get_received_values(), dtype=np.float64)
                if len(raw_vals) >= 8:
                    n = len(raw_vals)
                    fft_vals = np.fft.rfft(raw_vals)
                    freqs = np.fft.rfftfreq(n, d=self.space.dt)
                    amps = 2.0 * np.abs(fft_vals) / n
                    amps[0] /= 2.0
                    rx_fft_spec = amps[freqs <= 3.0e9].tolist()
            except Exception:
                rx_fft_spec = []

            rx_list.append({
                "id": rid, "x": float(r.x), "y": float(r.y),
                "fc": safe_call(r, "tuned_frequency", "fc", default=1.0e9),
                "rb": safe_call(r, "bit_rate", "rb", default=500.0e6),
                "rx_raw": list(r.get_received_values()),
                "rx_bpf": list(r.get_filtered_values()),
                "rx_mixed": list(r.get_mixed_values()),
                "rx_matched": list(r.get_baseband_values()),
                "rx_bits": list(r.get_bit_values()) if hasattr(r, "get_bit_values") else [],
                "spectrum": rx_fft_spec
            })

        obs_list = []
        for oid, op in self.observation_points.items():
            try:
                freqs, amps = op.compute_fft(window_type="hann")
                fft_data = amps[freqs <= 3.0e9].tolist() if len(amps) > 0 else []
                peak_f, peak_a = op.get_peak_frequency(window_type="hann")
            except Exception:
                fft_data = []
                peak_f, peak_a = 0.0, 0.0
            
            obs_list.append({
                "id": oid, "label": op.label,
                "x": float(op.x), "y": float(op.y),
                "waveform": list(op.signal_history),
                "spectrum": fft_data,
                "peak_freq_ghz": float(peak_f / 1e9),
                "peak_mag": float(peak_a)
            })

        links_list = []
        for lid, ldata in self.link_evaluators_dict.items():
            ev = ldata["evaluator"]
            ber_val = ev.get_bit_error_rate() if ev and ev.get_bit_error_rate() is not None else 0.0
            delay_val = ev.get_estimated_total_delay_seconds() * 1e9 if ev else 0.0
            links_list.append({
                "id": lid,
                "tx_id": ldata["tx_id"],
                "rx_id": ldata["rx_id"],
                "ber": float(ber_val),
                "delay_ns": float(delay_val),
                "bits_compared": ev.get_total_bits_compared() if ev else 0,
                "bit_errors": ev.get_bit_errors() if ev else 0,
            })

        return {
            "type": "telemetry",
            "running": self.running,
            "grid_width": self.resolution_x,
            "grid_height": self.resolution_y,
            "dt_multiplier": self.dt_multiplier,
            "time_ns": self.space.time * 1e9,
            "transmitters": tx_list,
            "receivers": rx_list,
            "observation_points": obs_list,
            "link_evaluators": links_list,
            "materials": self.materials_list,
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
        elif msg_type == "set_resolution":
            w = max(100, int(message.get("width", runtime.resolution_x)))
            h = max(100, int(message.get("height", runtime.resolution_y)))
            dt_mult = float(message.get("dt_multiplier", runtime.dt_multiplier))
            runtime.running = False
            runtime._build(res_x=w, res_y=h, dt_multiplier=dt_mult)
        elif msg_type == "set_view_mode":
            runtime.view_mode = message.get("mode", "field")
        elif msg_type == "add_transmitter":
            new_id = max(runtime.transmitters.keys()) + 1 if runtime.transmitters else 0
            runtime.add_transmitter(new_id, float(message.get("x", 2.0)), float(message.get("y", 7.0)))
        elif msg_type == "remove_transmitter":
            runtime.remove_transmitter(int(message.get("tx_id", 0)))
        
        elif msg_type == "set_transmitter_params":
            tx_id = int(message.get("tx_id", 0))
            if tx_id in runtime.transmitters:
                tx = runtime.transmitters[tx_id]
                if "fc" in message:
                    val = float(message["fc"])
                    if hasattr(tx, "carrier_frequency"): tx.carrier_frequency = val
                    if hasattr(tx, "fc"): tx.fc = val
                if "rb" in message:
                    val = float(message["rb"])
                    if hasattr(tx, "bit_rate"): tx.bit_rate = val
                    if hasattr(tx, "rb"): tx.rb = val
                
                amp_val = message.get("amp", message.get("amplitude", message.get("carrier_amplitude")))
                if amp_val is not None:
                    val = float(amp_val)
                    if hasattr(tx, "carrier_amplitude"): tx.carrier_amplitude = val
                    if hasattr(tx, "amplitude"): tx.amplitude = val
                    if hasattr(tx, "amp"): tx.amp = val

                if hasattr(tx, "update_parameters") and callable(tx.update_parameters):
                    tx.update_parameters()
                elif hasattr(tx, "_generate_waveform") and callable(tx._generate_waveform):
                    tx._generate_waveform()

                current_fc = safe_call(tx, "carrier_frequency", "fc", default=1.0e9)
                current_amp = safe_call(tx, "carrier_amplitude", "amplitude", "amp", default=2.0)
                print(f"[BACKEND] Updated Transmitter {tx_id} -> Freq: {current_fc} Hz, Amp: {current_amp} V")

        elif msg_type == "add_receiver":
            new_id = max(runtime.receivers.keys()) + 1 if runtime.receivers else 0
            runtime.add_receiver(new_id, float(message.get("x", 8.0)), float(message.get("y", 5.0)))
        elif msg_type == "remove_receiver":
            runtime.remove_receiver(int(message.get("rx_id", 0)))

        elif msg_type == "set_receiver_params":
            rx_id = int(message.get("rx_id", 0))
            if rx_id in runtime.receivers:
                rx = runtime.receivers[rx_id]
                if "fc" in message:
                    val = float(message["fc"])
                    if hasattr(rx, "tuned_frequency"): rx.tuned_frequency = val
                    if hasattr(rx, "fc"): rx.fc = val
                if "rb" in message:
                    val = float(message["rb"])
                    if hasattr(rx, "bit_rate"): rx.bit_rate = val
                    if hasattr(rx, "rb"): rx.rb = val
                print(f"[BACKEND] Updated Receiver {rx_id} -> Tuned Freq: {safe_call(rx, 'tuned_frequency', 'fc')} Hz, BitRate: {safe_call(rx, 'bit_rate', 'rb')} bps")

        elif msg_type == "add_observation_point":
            runtime.add_observation_point(float(message.get("x", 5.0)), float(message.get("y", 5.0)), label=message.get("label", "Probe"))
        elif msg_type == "remove_observation_point":
            runtime.remove_observation_point(int(message.get("oid", 0)))
        elif msg_type == "add_link_evaluator":
            tx_keys = list(runtime.transmitters.keys())
            rx_keys = list(runtime.receivers.keys())
            t_id = tx_keys[0] if tx_keys else 0
            r_id = rx_keys[0] if rx_keys else 0
            runtime.add_link_evaluator(t_id, r_id)
        elif msg_type == "remove_link_evaluator":
            runtime.remove_link_evaluator(int(message.get("lid", 0)))
        elif msg_type == "update_link_evaluator":
            runtime.update_link_evaluator_pairing(int(message.get("lid", 0)), int(message.get("tx_id", 0)), int(message.get("rx_id", 0)))
        elif msg_type == "move_tx":
            tx_id = int(message.get("tx_id", 0))
            if tx_id in runtime.transmitters:
                runtime.transmitters[tx_id].set_position(float(message["x"]), float(message["y"]))
        elif msg_type == "move_rx":
            rx_id = int(message.get("rx_id", 0))
            if rx_id in runtime.receivers:
                runtime.receivers[rx_id].set_position(float(message["x"]), float(message["y"]))
        elif msg_type == "move_obs":
            oid = int(message.get("oid", 0))
            if oid in runtime.observation_points:
                runtime.observation_points[oid].set_position(float(message["x"]), float(message["y"]))
        elif msg_type == "set_obs_window":
            oid = int(message.get("oid", 0))
            win_ns = float(message.get("window_ns", 15.0))
            if oid in runtime.observation_points:
                op = runtime.observation_points[oid]
                if hasattr(op, "set_buffer_duration"):
                    op.set_buffer_duration(win_ns * 1e-9)
        elif msg_type == "add_material":
            runtime.add_material(
                name=message.get("name", "concrete"),
                x_min=message.get("x_min"), 
                x_max=message.get("x_max"),
                y_min=message.get("y_min"), 
                y_max=message.get("y_max"),
                angle=float(message.get("angle", 0.0)),
                rel_perm=message.get("relative_permittivity"),
                rel_mu=message.get("relative_permeability"),
                cond=message.get("conductivity", 0.0),
            )
        elif msg_type == "remove_material":
            runtime.remove_material(int(message.get("mat_id", 0)))
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
                break

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_interval - elapsed))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receive_task.cancel()