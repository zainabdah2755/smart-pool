"""
Smart Pool API - Simulation Test Suite
=======================================
Tests the API with simulated sensor data BEFORE connecting real ESP32.
Run this after:
  1. python train_model_v2.py          → generates model .pkl files
  2. uvicorn smart_pool_api:app ...    → starts the API server

Usage:
  python test_simulation.py                   # runs against localhost:8000
  python test_simulation.py --host x.x.x.x   # custom host
"""

import argparse
import json
import sys
import time
import random
import requests


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

def get_base_url() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    args, _ = parser.parse_known_args()
    return f"http://{args.host}:{args.port}"


BASE = get_base_url()
DIVIDER = "─" * 60


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def post(endpoint: str, payload: dict) -> dict:
    resp = requests.post(f"{BASE}{endpoint}", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get(endpoint: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{BASE}{endpoint}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def print_result(label: str, result: dict):
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {label}")
    print(DIVIDER)
    print(f"  Status        : {result.get('status')}")
    print(f"  Health Score  : {result.get('health_score')}/100")
    print(f"  Algae Risk    : {result.get('algae_risk')}/100  [{result.get('algae_label')}]")
    cl = result.get("chlorine", {})
    print(f"  Chlorine Dose : {cl.get('dose_grams')} g  "
          f"(target {cl.get('free_chlorine_target')} ppm)")
    print(f"  Shock Required: {cl.get('shock_required')}")
    print(f"  Maintenance   : {result.get('needs_maintenance')}")

    alerts = result.get("maintenance_alerts", [])
    if alerts:
        print(f"\n  ⚠  ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"    [{a['level']:8s}] {a['message']}")
            print(f"             → {a['action']}")

    recs = result.get("recommendations", [])
    if recs:
        print(f"\n  📋 RECOMMENDATIONS:")
        for r in recs:
            print(f"    {r}")

    cl_rationale = cl.get("rationale", [])
    if cl_rationale:
        print(f"\n  🧪 CHLORINE RATIONALE:")
        for r in cl_rationale:
            print(f"    • {r}")

    weather = result.get("weather")
    if weather and not weather.get("error"):
        print(f"\n  🌤  WEATHER:")
        print(f"    {weather.get('description')}, "
              f"{weather.get('temperature')}°C, "
              f"UV {weather.get('uv_index')}")
        print(f"    Contamination risk : {weather.get('contamination_risk')}")
        print(f"    Evaporation        : {weather.get('evaporation_level')}")
        print(f"    Advisory           : {weather.get('maintenance_advisory')}")


# ─────────────────────────────────────────────
# TEST SCENARIOS
# ─────────────────────────────────────────────

SCENARIOS = [
    {
        "label": "✅ Normal — Perfect pool conditions",
        "payload": {
            "temp_eau": 26.5, "temp_air": 30.0, "humidite": 55.0,
            "ph": 7.3, "luminosite": 400.0, "ir": 0.05,
            "pool_volume_m3": 50.0,
        },
    },
    {
        "label": "⚠ Warning — High pH + moderate turbidity",
        "payload": {
            "temp_eau": 31.5, "temp_air": 35.0, "humidite": 70.0,
            "ph": 7.9, "luminosite": 750.0, "ir": 0.38,
            "pool_volume_m3": 50.0,
        },
    },
    {
        "label": "🚨 Danger — Extreme pH + high turbidity + hot water",
        "payload": {
            "temp_eau": 36.0, "temp_air": 40.0, "humidite": 80.0,
            "ph": 8.7, "luminosite": 900.0, "ir": 0.78,
            "pool_volume_m3": 50.0,
        },
    },
    {
        "label": "🌿 Algae Risk — Warm + humid + high UV",
        "payload": {
            "temp_eau": 29.5, "temp_air": 34.0, "humidite": 88.0,
            "ph": 8.1, "luminosite": 870.0, "ir": 0.52,
            "pool_volume_m3": 50.0,
        },
    },
    {
        "label": "🌡 Hot Day — High UV + evaporation",
        "payload": {
            "temp_eau": 33.5, "temp_air": 42.0, "humidite": 28.0,
            "ph": 7.5, "luminosite": 980.0, "ir": 0.08,
            "pool_volume_m3": 50.0,
        },
    },
    {
        "label": "🌍 With Weather (Marrakesh coords)",
        "payload": {
            "temp_eau": 28.0, "temp_air": 32.0, "humidite": 60.0,
            "ph": 7.4, "luminosite": 650.0, "ir": 0.15,
            "pool_volume_m3": 50.0,
            "latitude": 31.6295, "longitude": -7.9811,
        },
    },
]


# ─────────────────────────────────────────────
# STRESS TEST: continuous random sensor stream
# ─────────────────────────────────────────────

def stress_test(n_requests: int = 20, endpoint: str = "/predict/quick"):
    print(f"\n{'='*60}")
    print(f"  STRESS TEST — {n_requests} random requests → {endpoint}")
    print(DIVIDER)
    ok = 0
    errors = 0
    latencies = []

    for i in range(n_requests):
        payload = {
            "temp_eau":   random.uniform(20, 38),
            "temp_air":   random.uniform(18, 45),
            "humidite":   random.uniform(30, 95),
            "ph":         random.uniform(6.5, 8.8),
            "luminosite": random.uniform(0, 1023),
            "ir":         random.uniform(0, 0.9),
            "pool_volume_m3": 50.0,
        }
        t0 = time.time()
        try:
            result = post(endpoint, payload)
            latencies.append((time.time() - t0) * 1000)
            ok += 1
            sys.stdout.write(
                f"\r  [{i+1:3d}/{n_requests}] "
                f"status={result.get('status'):7s} "
                f"health={result.get('health_score'):5.1f} "
                f"latency={latencies[-1]:.0f}ms   "
            )
            sys.stdout.flush()
        except Exception as exc:
            errors += 1
            print(f"\n  ❌ Request {i+1} failed: {exc}")

    avg = sum(latencies) / len(latencies) if latencies else 0
    mx  = max(latencies) if latencies else 0
    print(f"\n\n  Results: {ok} OK  {errors} Errors")
    print(f"  Avg latency: {avg:.0f} ms  |  Max: {mx:.0f} ms")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"\n🏊 Smart Pool API Test Suite  →  {BASE}")
    print(DIVIDER)

    # Health check
    try:
        h = get("/health")
        print(f"  API Health : {h.get('status')}")
        print(f"  Models     : {'✅ Loaded' if h.get('models_loaded') else '⚠ Rule-based fallback'}")
        print(f"  Version    : {h.get('version')}")
    except requests.ConnectionError:
        print(f"\n  ❌ Cannot connect to {BASE}")
        print("  → Start the API first:  uvicorn smart_pool_api:app --reload")
        sys.exit(1)

    # Simulate endpoint test
    print(f"\n{DIVIDER}")
    print("  SIMULATE ENDPOINT TEST")
    for scenario in ["normal", "warning", "danger", "algae"]:
        sim = get("/simulate", {"scenario": scenario})
        print(f"  /simulate?scenario={scenario:<8} → temp_eau={sim['temp_eau']:.1f}  "
              f"ph={sim['ph']:.2f}  ir={sim['ir']:.2f}")

    # Full scenario tests
    for scene in SCENARIOS:
        result = post("/predict", scene["payload"])
        print_result(scene["label"], result)

    # Stress test
    stress_test(n_requests=30)

    print(f"\n{'='*60}")
    print("  ✅ All tests complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
