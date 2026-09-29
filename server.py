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

def clamp_window_ns(value):
    """FFT / scope window from the UI, kept to a sane range."""
    return min(500.0, max(2.0, float(value)))


def validate_link_params(fc, rb):
    """Reject carrier / bit-rate combinations the receiver front end cannot
    be designed for (band-pass low edge = fc - rb must stay positive)."""
    if fc <= 0 or rb <= 0:
        raise ValueError("Carrier frequency and bit rate must be positive.")
    if rb >= fc:
        raise ValueError(
            f"Bit rate ({rb / 1e6:.0f} Mbps) must be below the carrier "
            f"frequency ({fc / 1e9:.2f} GHz)."
        )

PRESETS = {
    "los": {
        "name": "Line-of-Sight",

        "noise_level": 0.0,

        "transmitters": [
            {
                "id": 0,
                "x": 3.0,
                "y": 5.0,
                "fc": 1.5e9,
                "rb": 400.0e6,
                "amp": 2.0,
                "window_duration": 20e-9,
            }
        ],

        "receivers": [
            {
                "id": 0,
                "x": 7.0,
                "y": 5.0,
                "fc": 1.5e9,
                "bit_rate": 400.0e6,
                "observation_window": 20e-9,
            }
        ],

        "observation_points": [
            {
                "x": 5.0,
                "y": 3.0,
                "label": "Grid Probe 0",
            }
        ],

        "link_evaluators": [
            {
                "tx_id": 0,
                "rx_id": 0,
            }
        ],

        "materials": [],
    },
        "empty": {
        "name": "Empty Canvas",
        "noise_level": 0.0,

        "transmitters": [],
        "receivers": [],
        "observation_points": [],
        "link_evaluators": [],
        "materials": [],
    },
}

class SimulationRuntime:
    def __init__(self):
        self.resolution_x = 1000
        self.resolution_y = 1000
        self.width = 10.0
        self.height = 10.0
        self.dt_multiplier = 0.25
        self.noise_level = 0.0
        self.steps_per_frame = 6
        self.target_fps = 60
        self.running = False
        self.view_mode = "field"

        self.transmitters = {}
        self.receivers = {}
        self.observation_points = {}
        self.link_evaluators_dict = {}
        
        self.next_obs_id = 0
        self.next_link_eval_id = 0

        self.materials_list = []

        self.current_preset = "los"

        self._build(preset=self.current_preset)

    def _build(
        self,
        res_x=None,
        res_y=None,
        dt_multiplier=None,
        noise_level=None,
        preset=None,
    ):
        if res_x is not None:
            self.resolution_x = max(100, int(res_x))

        if res_y is not None:
            self.resolution_y = max(100, int(res_y))

        if dt_multiplier is not None:
            # > 1.0 violates the Courant condition.
            self.dt_multiplier = min(1.0, max(0.01, float(dt_multiplier)))

        # If a preset was explicitly supplied, use it.
        # Otherwise keep the currently selected preset.
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset '{preset}'.")
            self.current_preset = preset

        config = PRESETS[self.current_preset]

        # Preset provides the default noise.
        # An explicitly supplied noise_level overrides the preset.
        if noise_level is not None:
            self.noise_level = max(0.0, float(noise_level))
        else:
            self.noise_level = max(
                0.0,
                float(config.get("noise_level", 0.0))
            )

        self.space = SimulationSpace(
            width=self.width,
            height=self.height,
            resolution_x=self.resolution_x,
            resolution_y=self.resolution_y,
            dt_stability_multiplier=self.dt_multiplier,
        )

        self.transmitters.clear()
        self.receivers.clear()
        self.observation_points.clear()
        self.link_evaluators_dict.clear()
        self.materials_list.clear()

        self.next_obs_id = 0
        self.next_link_eval_id = 0

        # ---------------------------------------------------------
        # TRANSMITTERS
        # ---------------------------------------------------------
        for tx_config in config.get("transmitters", []):
            tx_config = dict(tx_config)
            tx_id = tx_config.pop("id")

            self.add_transmitter(
                tx_id=tx_id,
                **tx_config
            )

        # ---------------------------------------------------------
        # RECEIVERS
        # ---------------------------------------------------------
        for rx_config in config.get("receivers", []):
            rx_config = dict(rx_config)
            rx_id = rx_config.pop("id")

            self.add_receiver(
                rx_id=rx_id,
                **rx_config
            )

        # ---------------------------------------------------------
        # OBSERVATION POINTS
        # ---------------------------------------------------------
        for obs_config in config.get("observation_points", []):
            self.add_observation_point(**obs_config)

        # ---------------------------------------------------------
        # WAVE SOLVER
        # ---------------------------------------------------------
        self.wave_solver = WaveSolver(
            self.space,
            noise_level=self.noise_level,
        )

        # ---------------------------------------------------------
        # MATERIALS
        # ---------------------------------------------------------
        for material_config in config.get("materials", []):
            self.add_material(
                **material_config,
                refresh=False,
            )

        # Refresh only once after all materials are applied.
        if config.get("materials"):
            self.wave_solver.refresh()

        # ---------------------------------------------------------
        # LINK EVALUATORS
        # ---------------------------------------------------------
        for link_config in config.get("link_evaluators", []):
            self.add_link_evaluator(**link_config)

        self.space.set_running(True)

    def add_transmitter(self, tx_id, x, y, fc=1.0e9, rb=500.0e6, amp=2.0, window_duration=20e-9):
        tx = Transmitter(
            self.space, x=float(x), y=float(y),
            carrier_frequency=float(fc), carrier_amplitude=float(amp),
            bit_rate=float(rb), window_duration=float(window_duration),
            fft_window=float(window_duration),
        )
        self.transmitters[tx_id] = tx

    def remove_transmitter(self, tx_id):
        if tx_id in self.transmitters:
            del self.transmitters[tx_id]
            to_del = [lid for lid, data in self.link_evaluators_dict.items() if data["tx_id"] == tx_id]
            for lid in to_del:
                del self.link_evaluators_dict[lid]

    def add_receiver(self, rx_id, x, y, fc=1.0e9, bit_rate=500.0e6, observation_window=20e-9):
        rx = Receiver(
            self.space, x=float(x), y=float(y),
            tuned_frequency=float(fc), bit_rate=float(bit_rate), observation_window=float(observation_window),
            fft_window=float(observation_window),
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
            buffer_duration=20e-9, label=label or f"Obs Point {oid}"
        )
        self.observation_points[oid] = op
        return oid

    def remove_observation_point(self, oid):
        if oid in self.observation_points:
            del self.observation_points[oid]

    def reset_links_for(self, tx_id=None, rx_id=None):
        """Clear BER / sync state of every evaluator that uses the given
        transmitter or receiver (their bit streams just restarted)."""
        for data in self.link_evaluators_dict.values():
            uses_tx = tx_id is not None and data["tx_id"] == tx_id
            uses_rx = rx_id is not None and data["rx_id"] == rx_id
            if (uses_tx or uses_rx) and data["evaluator"]:
                data["evaluator"].reset()

    def load_scenario(self, name):
        """Load and build a preset from PRESETS."""

        if name not in PRESETS:
            raise ValueError(f"Unknown preset '{name}'.")

        self.running = False
        self._build(preset=name)

    def add_link_evaluator(self, tx_id, rx_id):
        lid = self.next_link_eval_id
        self.next_link_eval_id += 1

        tx = self.transmitters.get(tx_id)
        rx = self.receivers.get(rx_id)

        ev = None

        if tx and rx:
            ev = LinkEvaluator(
                transmitter=tx,
                receiver=rx,
                speed_of_light=3.0e8,
            )

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
                transmitter=tx,
                receiver=rx,
                speed_of_light=3.0e8,
            )

            self.link_evaluators_dict[lid] = {
                "tx_id": tx_id,
                "rx_id": rx_id,
                "evaluator": ev
            }
            
    def add_material(self, name, x_min=None, x_max=None, y_min=None, y_max=None,
                     angle=0.0, rel_perm=None, rel_mu=None, cond=None,
                     refresh=True):
        if x_min is None or x_max is None or y_min is None or y_max is None:
            idx = len(self.materials_list)
            w_box, h_box = 0.4, 4.0
            x_min = 4.5 + (idx * 1.2) % 4.0
            x_max = x_min + w_box
            y_min = 2.0
            y_max = y_min + h_box

        # Built-in names use the database values unless overridden;
        # any other name is a custom material and must carry all three
        # properties (Material raises a clear error otherwise).
        mat = Material(
            simulation_space=self.space,
            name=name or "concrete",
            x_min=float(x_min), x_max=float(x_max),
            y_min=float(y_min), y_max=float(y_max),
            relative_permittivity=None if rel_perm is None else float(rel_perm),
            relative_permeability=None if rel_mu is None else float(rel_mu),
            conductivity=None if cond is None else float(cond),
        )
        mat.apply()

        if refresh and hasattr(self.wave_solver, "refresh"):
            self.wave_solver.refresh()

        # Report what the solver is actually using, not what was requested.
        self.materials_list.append({
            "id": len(self.materials_list),
            "name": mat.get_name(),
            "x_min": mat.x_min, "x_max": mat.x_max,
            "y_min": mat.y_min, "y_max": mat.y_max,
            "angle": float(angle),
            "relative_permittivity": mat.get_relative_permittivity(),
            "relative_permeability": mat.get_relative_permeability(),
            "conductivity": mat.get_conductivity(),
        })

    def remove_material(self, mat_id):
        remaining = [
            m for m in self.materials_list
            if m.get("id") != mat_id
        ]

        self.space.reset_materials()
        self.materials_list.clear()

        for m in remaining:
            self.add_material(
                name=m["name"],
                x_min=m["x_min"], x_max=m["x_max"],
                y_min=m["y_min"], y_max=m["y_max"],
                angle=m["angle"],
                rel_perm=m["relative_permittivity"],
                rel_mu=m["relative_permeability"],
                cond=m["conductivity"],
                refresh=False,   # one refresh at the end, not one per material
            )

        if hasattr(self.wave_solver, "refresh"):
            self.wave_solver.refresh()

    def clear_materials(self):
        self.space.reset_materials()
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
            tx_freqs, tx_amps, tx_df, tx_res = tx.compute_bpsk_fft()

            if len(tx_freqs) > 0:
                keep = tx_freqs <= 3.0e9
                fft_spec = tx_amps[keep].tolist()
            else:
                fft_spec = []

            tx_list.append({
                "id": tid, "x": float(tx.x), "y": float(tx.y),
                "fc": tx.get_carrier_frequency(),
                "rb": tx.get_bit_rate(),
                "amp": tx.get_carrier_amplitude(),
                "window_ns": tx.get_window_duration() * 1e9,
                "spectrum_df_hz": tx_df,
                "spectrum_res_hz": tx_res,
                "symbols": list(tx.get_bit_values()) if hasattr(tx, "get_bit_values") else [],
                "shaped": list(tx.get_shaped_values()) if hasattr(tx, "get_shaped_values") else [],
                "carrier": list(tx.get_carrier_values()) if hasattr(tx, "get_carrier_values") else [],
                "bpsk": list(tx.get_bpsk_values()) if hasattr(tx, "get_bpsk_values") else [],
                "spectrum": fft_spec,
            })

        rx_list = []
        for rid, r in self.receivers.items():
            rx_freqs, rx_amps, rx_df, rx_res = r.compute_filtered_fft()

            if len(rx_freqs) > 0:
                keep = rx_freqs <= 3.0e9
                rx_fft_spec = rx_amps[keep].tolist()
            else:
                rx_fft_spec = []

            rx_list.append({
                "id": rid, "x": float(r.x), "y": float(r.y),
                "fc": r.get_tuned_frequency(),
                "rb": r.get_bit_rate(),
                "window_ns": r.get_observation_window() * 1e9,
                "spectrum_df_hz": rx_df,
                "spectrum_res_hz": rx_res,
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
                freqs, amps, obs_df, obs_res = op.compute_fft()

                fft_data = (
                    amps[freqs <= 3.0e9].tolist()
                    if len(amps) > 0
                    else []
                )

                peak_f, peak_a = op.get_peak_frequency()

            except Exception:
                fft_data = []
                obs_df = 0.0
                obs_res = 0.0
                peak_f, peak_a = 0.0, 0.0
            
            obs_list.append({
                "id": oid, "label": op.label,
                "x": float(op.x), "y": float(op.y),
                "waveform": list(op.signal_history),
                "spectrum": fft_data,
                "peak_freq_ghz": float(peak_f / 1e9),
                "peak_mag": float(peak_a),
                "spectrum_df_hz": obs_df,
                "spectrum_res_hz": obs_res,
                "window_ns": op.buffer_duration * 1e9
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
                "synced": bool(ev.sync_found) if ev else False,
                "inverted": bool(ev.inverted) if ev else False,
            })

        preset_list = [
            {
                "id": preset_id,
                "name": config.get("name", preset_id),
            }
            for preset_id, config in PRESETS.items()
        ]

        return {
            "type": "telemetry",
            "running": self.running,
            "grid_width": self.resolution_x,
            "grid_height": self.resolution_y,
            "physical_width": self.width,
            "physical_height": self.height,
            "computational_width": self.space.computational_width,
            "computational_height": self.space.computational_height,
            "absorbing_layer_thickness": self.space.absorbing_layer_thickness,
            "dt_multiplier": self.dt_multiplier,
            "time_ns": self.space.time * 1e9,
            "dt": self.space.dt,
            "transmitters": tx_list,
            "receivers": rx_list,
            "observation_points": obs_list,
            "link_evaluators": links_list,
            "materials": self.materials_list,
            "view_mode": self.view_mode,
            "noise_level": self.wave_solver.get_noise_level(),
            "presets": preset_list,
            "current_preset": self.current_preset,
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
    """Apply one UI message to the runtime.

    Returns an error string when the request was rejected (the WebSocket
    loop forwards it to the browser), otherwise None.
    """
    msg_type = message.get("type")
    try:
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
            noise_level = max(
                0.0,
                float(message.get("noise_level", runtime.noise_level))
            )

            runtime.running = False

            runtime._build(
                res_x=w,
                res_y=h,
                dt_multiplier=dt_mult,
                noise_level=noise_level
            )
        elif msg_type == "load_scenario":
            runtime.load_scenario(message.get("name", "los"))
        elif msg_type == "set_view_mode":
            runtime.view_mode = message.get("mode", "field")
        elif msg_type == "set_noise":
            runtime.noise_level = max(
                0.0,
                float(message.get("level", 0.0))
            )
            runtime.wave_solver.set_noise_level(runtime.noise_level)

        # ---------------- transmitters ----------------
        elif msg_type == "add_transmitter":
            new_id = max(runtime.transmitters.keys()) + 1 if runtime.transmitters else 0
            runtime.add_transmitter(new_id, float(message.get("x", 2.0)), float(message.get("y", 7.0)))
        elif msg_type == "remove_transmitter":
            runtime.remove_transmitter(int(message.get("tx_id", 0)))
        elif msg_type == "set_transmitter_params":
            tx_id = int(message.get("tx_id", 0))
            tx = runtime.transmitters.get(tx_id)
            if tx is not None:
                new_fc = float(message["fc"]) if "fc" in message else tx.get_carrier_frequency()
                new_rb = float(message["rb"]) if "rb" in message else tx.get_bit_rate()
                validate_link_params(new_fc, new_rb)

                if new_fc != tx.get_carrier_frequency():
                    tx.set_carrier_frequency(new_fc)

                # set_bit_rate restarts the bit stream (and preamble), so only
                # call it when the rate really changed - not on every slider tick.
                if new_rb != tx.get_bit_rate():
                    tx.set_bit_rate(new_rb)
                    runtime.reset_links_for(tx_id=tx_id)

                amp_val = message.get("amp", message.get("amplitude", message.get("carrier_amplitude")))
                if amp_val is not None:
                    tx.set_carrier_amplitude(float(amp_val))

                if "window_ns" in message:
                    win = clamp_window_ns(message["window_ns"]) * 1e-9
                    tx.set_window_duration(win)     # time-domain traces
                    tx.set_fft_window(win)          # FFT input buffer

        # ---------------- receivers ----------------
        elif msg_type == "add_receiver":
            new_id = max(runtime.receivers.keys()) + 1 if runtime.receivers else 0
            runtime.add_receiver(new_id, float(message.get("x", 8.0)), float(message.get("y", 5.0)))
        elif msg_type == "remove_receiver":
            runtime.remove_receiver(int(message.get("rx_id", 0)))
        elif msg_type == "set_receiver_params":
            rx_id = int(message.get("rx_id", 0))
            rx = runtime.receivers.get(rx_id)
            if rx is not None:
                new_fc = float(message["fc"]) if "fc" in message else rx.get_tuned_frequency()
                new_rb = float(message["rb"]) if "rb" in message else rx.get_bit_rate()
                validate_link_params(new_fc, new_rb)

                # Use the real setters: they redesign the band-pass filter,
                # matched filter, Costas loop and Gardner loop. Assigning the
                # attributes directly left all of those on the old design.
                # Apply in an order that keeps (fc - rb) > 0 at every step.
                if new_fc >= rx.get_tuned_frequency():
                    steps = [("fc", new_fc), ("rb", new_rb)]
                else:
                    steps = [("rb", new_rb), ("fc", new_fc)]

                changed = False
                for key, val in steps:
                    if key == "fc" and val != rx.get_tuned_frequency():
                        rx.set_tuned_frequency(val)
                        changed = True
                    elif key == "rb" and val != rx.get_bit_rate():
                        rx.set_bit_rate(val)
                        changed = True

                if changed:
                    rx.demodulated_bits.clear()
                    rx.demodulated_bit_times.clear()
                    runtime.reset_links_for(rx_id=rx_id)

                if "window_ns" in message:
                    win = clamp_window_ns(message["window_ns"]) * 1e-9
                    rx.set_observation_window(win)  # time-domain traces + FFT input
                    rx.set_fft_window(win)

        # ---------------- observation points ----------------
        elif msg_type == "add_observation_point":
            runtime.add_observation_point(float(message.get("x", 5.0)), float(message.get("y", 5.0)), label=message.get("label", "Probe"))
        elif msg_type == "remove_observation_point":
            runtime.remove_observation_point(int(message.get("oid", 0)))
        elif msg_type == "set_obs_window":
            oid = int(message.get("oid", 0))
            op = runtime.observation_points.get(oid)
            if op is not None:
                op.set_fft_window(clamp_window_ns(message.get("window_ns", 15.0)) * 1e-9)

        # ---------------- link evaluators ----------------
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

        # ---------------- dragging ----------------
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

        # ---------------- materials ----------------
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
                cond=message.get("conductivity"),
            )
        elif msg_type == "remove_material":
            runtime.remove_material(int(message.get("mat_id", 0)))
        elif msg_type in ("clear_walls", "clear_materials"):
            runtime.clear_materials()

    except Exception as e:
        print(f"Error handling control message {message}: {e}")
        return f"{msg_type}: {e}"
    return None


@app.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_json(runtime.status())
    except Exception:
        return

    pending_errors = []

    async def receive_loop():
        try:
            while True:
                msg = await websocket.receive_json()
                error = handle_control_message(msg)
                if error:
                    pending_errors.append(error)
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
                while pending_errors:
                    await websocket.send_json({"type": "error", "message": pending_errors.pop(0)})
            except Exception:
                break

            elapsed = time.perf_counter() - loop_start
            await asyncio.sleep(max(0.001, frame_interval - elapsed))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        receive_task.cancel()