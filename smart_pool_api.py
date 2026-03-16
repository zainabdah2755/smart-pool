"""
Smart Pool AI API v2  —  FastAPI Application
=============================================
Endpoints:
  POST /predict          — Full AI analysis (main endpoint)
  POST /predict/quick    — Lightweight prediction (no weather)
  GET  /health           — API health check
  GET  /simulate         — Generate simulated sensor payload for testing
  GET  /docs             — Auto-generated Swagger UI (built-in FastAPI)

Run with:
  uvicorn smart_pool_api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import math
import random
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from chlorine_optimizer import calculate_chlorine_dose
from weather_service import fetch_weather_sync

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

try:
    model_status      = joblib.load("model_status.pkl")
    status_encoder    = joblib.load("model_status_encoder.pkl")
    model_health      = joblib.load("model_health.pkl")
    model_algae       = joblib.load("model_algae.pkl")
    model_maintenance = joblib.load("model_maintenance.pkl")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    FEATURE_NAMES = meta["feature_names"]
    MODELS_LOADED = True
    print("✅ All models loaded successfully")
except FileNotFoundError as exc:
    print(f"⚠  Model files not found ({exc}). Run train_model_v2.py first.")
    MODELS_LOADED = False
    FEATURE_NAMES = [
        "temp_eau","temp_air","humidite","ph",
        "luminosite","ir",
        "temp_delta","heat_index","ph_deviation",
        "clarity_score","uv_index_est",
    ]


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Smart Pool AI API",
    version="2.0",
    description="AI-powered pool monitoring, chlorine optimization, algae prediction & maintenance alerts",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────

class SensorData(BaseModel):
    # Raw ESP32 sensor readings
    temp_eau:   float = Field(..., ge=-10, le=50,   description="Water temperature (°C)")
    temp_air:   float = Field(..., ge=-20, le=60,   description="Air temperature (°C)")
    humidite:   float = Field(..., ge=0,   le=100,  description="Humidity (%)")
    ph:         float = Field(..., ge=0,   le=14,   description="Water pH")
    luminosite: float = Field(..., ge=0,   le=1023, description="Light level (LDR 0-1023)")
    ir:         float = Field(..., ge=0,   le=1,    description="Turbidity (IR 0=clear, 1=opaque)")

    # Optional location for weather integration
    latitude:   Optional[float] = Field(None, description="GPS latitude for weather")
    longitude:  Optional[float] = Field(None, description="GPS longitude for weather")
    pool_volume_m3: float = Field(50.0, ge=1, le=5000, description="Pool volume in m³")
    current_free_cl: Optional[float] = Field(None, ge=0, le=10, description="Measured free Cl (ppm)")

    @field_validator("ph")
    @classmethod
    def ph_range(cls, v: float) -> float:
        if not (4.0 <= v <= 10.0):
            raise ValueError("pH must be between 4.0 and 10.0")
        return v


class MaintenanceAlert(BaseModel):
    level: str         # Info / Warning / Critical
    message: str
    action: str


class ChlorineInfo(BaseModel):
    dose_grams: int
    free_chlorine_target: float
    shock_required: bool
    rationale: list[str]


class WeatherInfo(BaseModel):
    temperature: Optional[float]
    humidity: Optional[float]
    uv_index: Optional[float]
    description: Optional[str]
    chlorine_extra_ppm: Optional[float]
    evaporation_level: Optional[str]
    contamination_risk: Optional[str]
    cover_recommendation: Optional[bool]
    maintenance_advisory: Optional[str]
    forecast_summary: Optional[list[str]]
    error: Optional[str]


class PredictionResponse(BaseModel):
    # Core pool status
    status:        str            # Normal / Warning / Danger
    health_score:  float          # 0-100
    # Algae
    algae_risk:    float          # 0-100
    algae_label:   str            # Low / Moderate / High / Critical
    # Chlorine
    chlorine:      ChlorineInfo
    # Maintenance
    needs_maintenance: bool
    maintenance_alerts: list[MaintenanceAlert]
    # Recommendations
    recommendations: list[str]
    # Weather (optional)
    weather:       Optional[WeatherInfo]
    # Meta
    model_version: str


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (mirrors training)
# ─────────────────────────────────────────────

def engineer(data: SensorData) -> list[float]:
    temp_delta   = data.temp_eau - data.temp_air
    heat_index   = (
        data.temp_air
        + 0.33 * (data.humidite / 100 * 6.105
                  * math.exp(17.27 * data.temp_air / (237.7 + data.temp_air)))
        - 4
    )
    ph_deviation  = abs(data.ph - 7.4)
    clarity_score = max(0.0, 1.0 - data.ir)
    uv_index_est  = data.luminosite / 1023 * 11

    return [
        data.temp_eau, data.temp_air, data.humidite, data.ph,
        data.luminosite, data.ir,
        temp_delta, heat_index, ph_deviation, clarity_score, uv_index_est,
    ]


# ─────────────────────────────────────────────
# MAINTENANCE ALERT GENERATOR
# ─────────────────────────────────────────────

def generate_alerts(data: SensorData, algae_risk: float) -> list[MaintenanceAlert]:
    alerts: list[MaintenanceAlert] = []

    if data.ph < 6.8:
        alerts.append(MaintenanceAlert(
            level="Critical", message=f"pH critically low ({data.ph:.2f})",
            action="Add pH Increaser (sodium carbonate) immediately",
        ))
    elif data.ph < 7.2:
        alerts.append(MaintenanceAlert(
            level="Warning", message=f"pH low ({data.ph:.2f})",
            action="Add pH Increaser — target 7.2–7.6",
        ))
    elif data.ph > 8.5:
        alerts.append(MaintenanceAlert(
            level="Critical", message=f"pH critically high ({data.ph:.2f})",
            action="Add pH Reducer (sodium bisulfate) immediately",
        ))
    elif data.ph > 7.8:
        alerts.append(MaintenanceAlert(
            level="Warning", message=f"pH high ({data.ph:.2f})",
            action="Add pH Reducer — target 7.2–7.6",
        ))

    if data.temp_eau > 35:
        alerts.append(MaintenanceAlert(
            level="Critical", message=f"Water temperature dangerous ({data.temp_eau:.1f}°C)",
            action="Pause pool use; cool water or wait for nighttime cooling",
        ))
    elif data.temp_eau > 30:
        alerts.append(MaintenanceAlert(
            level="Warning", message=f"Water temperature elevated ({data.temp_eau:.1f}°C)",
            action="Increase chlorine frequency; consider pool shade",
        ))

    if data.ir > 0.7:
        alerts.append(MaintenanceAlert(
            level="Critical", message="Severe turbidity — water is very cloudy",
            action="Run filter 24h, backwash, apply clarifier + shock dose",
        ))
    elif data.ir > 0.4:
        alerts.append(MaintenanceAlert(
            level="Warning", message="Moderate turbidity detected",
            action="Run filter continuously, add clarifier",
        ))

    if algae_risk >= 75:
        alerts.append(MaintenanceAlert(
            level="Critical", message=f"Algae bloom imminent (risk {algae_risk:.0f}/100)",
            action="Shock dose chlorine (10× normal), brush walls, run filter 48h",
        ))
    elif algae_risk >= 50:
        alerts.append(MaintenanceAlert(
            level="Warning", message=f"Elevated algae risk ({algae_risk:.0f}/100)",
            action="Increase chlorine, add algaecide, check circulation",
        ))

    return alerts


# ─────────────────────────────────────────────
# RECOMMENDATIONS GENERATOR
# ─────────────────────────────────────────────

def generate_recommendations(
    data: SensorData,
    status: str,
    algae_risk: float,
    health_score: float,
) -> list[str]:
    recs: list[str] = []

    if 7.2 <= data.ph <= 7.6:
        recs.append("✅ pH is in the ideal range (7.2–7.6) — no adjustment needed")
    else:
        recs.append(f"⚠ Adjust pH toward ideal range 7.2–7.6 (current: {data.ph:.2f})")

    if data.ir < 0.1:
        recs.append("✅ Water clarity is excellent")
    elif data.ir < 0.3:
        recs.append("💧 Run filter 2 extra hours today to improve clarity")

    if 26 <= data.temp_eau <= 30:
        recs.append("✅ Water temperature is ideal for swimming")
    elif data.temp_eau > 30:
        recs.append("🌡 High water temp: dose chlorine in the evening to reduce UV degradation")

    if algae_risk < 30:
        recs.append("✅ Algae risk is low — maintain regular weekly maintenance")
    elif algae_risk < 60:
        recs.append("🌿 Add algaecide this week as a precaution")
    else:
        recs.append("🚨 High algae risk — immediate treatment required")

    if health_score >= 85:
        recs.append("🏊 Pool is ready for use — water quality is excellent")
    elif health_score >= 65:
        recs.append("🔧 Pool quality is acceptable — schedule maintenance this week")
    else:
        recs.append("🚫 Pool quality is poor — do not use until treated")

    if data.luminosite > 700:
        recs.append("☀️ High UV day: test chlorine levels again at noon and evening")

    return recs


# ─────────────────────────────────────────────
# FALLBACK PREDICTION (if models not loaded)
# ─────────────────────────────────────────────

def rule_based_predict(data: SensorData):
    """Safety-net rule engine used when ML models aren't loaded."""
    ph_ok   = 7.0 <= data.ph <= 7.8
    temp_ok = data.temp_eau <= 32
    ir_ok   = data.ir <= 0.4

    if not ph_ok or not temp_ok or data.ir > 0.7:
        status = "Danger"; health = 35.0
    elif not ir_ok or data.ph > 7.6 or data.temp_eau > 30:
        status = "Warning"; health = 65.0
    else:
        status = "Normal";  health = 88.0

    algae = max(0, (data.temp_eau - 24) * 3 + (data.ph - 7.8) * 15 + data.ir * 30)
    algae = min(100.0, algae)
    return status, float(health), float(algae), bool(health < 70)


# ─────────────────────────────────────────────
# MAIN PREDICTION ENDPOINT
# ─────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorData):
    """
    Full AI-powered pool analysis.
    Includes ML predictions, chlorine dosing, algae risk,
    maintenance alerts, recommendations and optional weather data.
    """
    # ── Safety hard-limits before ML ─────────────────────────────
    if data.ph < 6.0 or data.ph > 9.5:
        raise HTTPException(422, detail=f"pH value {data.ph} is outside sensor range")

    # ── ML Predictions ────────────────────────────────────────────
    if MODELS_LOADED:
        feats = np.array([engineer(data)])
        status_enc   = model_status.predict(feats)[0]
        status       = status_encoder.inverse_transform([status_enc])[0]
        health_score = float(np.clip(model_health.predict(feats)[0], 0, 100))
        algae_risk   = float(np.clip(model_algae.predict(feats)[0], 0, 100))
        needs_maint  = bool(model_maintenance.predict(feats)[0])
    else:
        status, health_score, algae_risk, needs_maint = rule_based_predict(data)

    # ── Algae label ───────────────────────────────────────────────
    if algae_risk >= 75:  algae_label = "Critical"
    elif algae_risk >= 50: algae_label = "High"
    elif algae_risk >= 25: algae_label = "Moderate"
    else:                  algae_label = "Low"

    # ── Weather fetch ─────────────────────────────────────────────
    weather_info: Optional[WeatherInfo] = None
    weather_algae_factor = 0.0

    if data.latitude is not None and data.longitude is not None:
        wd = fetch_weather_sync(data.latitude, data.longitude)
        if wd.error:
            weather_info = WeatherInfo(
                temperature=None, humidity=None, uv_index=None,
                description=None, chlorine_extra_ppm=None,
                evaporation_level=None, contamination_risk=None,
                cover_recommendation=None, maintenance_advisory=None,
                forecast_summary=None, error=wd.error,
            )
        else:
            c = wd.current
            imp = wd.impact
            weather_info = WeatherInfo(
                temperature=c.temperature_2m,
                humidity=c.relative_humidity_2m,
                uv_index=c.uv_index,
                description=c.weather_description,
                chlorine_extra_ppm=imp.chlorine_extra_ppm,
                evaporation_level=imp.evaporation_level,
                contamination_risk=imp.contamination_risk,
                cover_recommendation=imp.cover_recommendation,
                maintenance_advisory=imp.maintenance_advisory,
                forecast_summary=imp.forecast_summary,
                error=None,
            )
            weather_algae_factor = imp.algae_weather_factor
            # Boost algae risk with weather factor
            algae_risk = float(min(100, algae_risk + weather_algae_factor * 20))

    # ── Chlorine dose ─────────────────────────────────────────────
    extra_cl = 0.0
    if weather_info and weather_info.chlorine_extra_ppm:
        extra_cl = weather_info.chlorine_extra_ppm

    cl_result = calculate_chlorine_dose(
        ph=data.ph,
        temp_eau=data.temp_eau,
        ir=data.ir,
        humidite=data.humidite,
        luminosite=data.luminosite,
        algae_risk=algae_risk,
        pool_volume_m3=data.pool_volume_m3,
        current_free_cl=data.current_free_cl,
    )
    # Add weather-driven extra dose
    if extra_cl > 0:
        bonus_grams = int(math.ceil(extra_cl * data.pool_volume_m3 / 0.9))
        cl_result.dose_grams += bonus_grams
        cl_result.rationale.append(
            f"Weather impact: +{bonus_grams}g extra due to weather conditions"
        )

    # ── Maintenance alerts ────────────────────────────────────────
    alerts = generate_alerts(data, algae_risk)
    if cl_result.shock_required:
        alerts.append(MaintenanceAlert(
            level="Warning",
            message="Breakpoint shock treatment recommended",
            action=f"Add {cl_result.dose_grams * 10}g shock chlorine (10× normal dose)",
        ))

    # ── Recommendations ───────────────────────────────────────────
    recs = generate_recommendations(data, status, algae_risk, health_score)

    return PredictionResponse(
        status=status,
        health_score=round(health_score, 1),
        algae_risk=round(algae_risk, 1),
        algae_label=algae_label,
        chlorine=ChlorineInfo(
            dose_grams=cl_result.dose_grams,
            free_chlorine_target=cl_result.free_chlorine_target,
            shock_required=cl_result.shock_required,
            rationale=cl_result.rationale,
        ),
        needs_maintenance=needs_maint or len([a for a in alerts if a.level == "Critical"]) > 0,
        maintenance_alerts=alerts,
        recommendations=recs,
        weather=weather_info,
        model_version="2.0",
    )


# ─────────────────────────────────────────────
# QUICK PREDICT (no weather, minimal compute)
# ─────────────────────────────────────────────

@app.post("/predict/quick")
def predict_quick(data: SensorData):
    """Lightweight prediction without weather fetch — lower latency."""
    # Temporarily null out coords
    data.latitude = None
    data.longitude = None
    return predict(data)


# ─────────────────────────────────────────────
# SIMULATE ENDPOINT (for testing)
# ─────────────────────────────────────────────

@app.get("/simulate")
def simulate(scenario: str = "normal"):
    """
    Returns a realistic simulated sensor payload.
    Use ?scenario=normal | warning | danger | algae | hot
    """
    scenarios = {
        "normal": dict(
            temp_eau=26.5, temp_air=30.0, humidite=55.0,
            ph=7.3, luminosite=400.0, ir=0.05,
        ),
        "warning": dict(
            temp_eau=31.5, temp_air=35.0, humidite=70.0,
            ph=7.9, luminosite=750.0, ir=0.35,
        ),
        "danger": dict(
            temp_eau=36.0, temp_air=40.0, humidite=80.0,
            ph=8.6, luminosite=900.0, ir=0.75,
        ),
        "algae": dict(
            temp_eau=29.0, temp_air=33.0, humidite=85.0,
            ph=8.1, luminosite=850.0, ir=0.55,
        ),
        "hot": dict(
            temp_eau=34.0, temp_air=42.0, humidite=30.0,
            ph=7.5, luminosite=950.0, ir=0.10,
        ),
    }
    base = scenarios.get(scenario, scenarios["normal"])
    # Add small random noise for realism
    base["temp_eau"]   += random.uniform(-0.3, 0.3)
    base["ph"]         += random.uniform(-0.05, 0.05)
    base["ir"]         += random.uniform(-0.01, 0.01)
    base["pool_volume_m3"] = 50.0
    return base


# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": MODELS_LOADED,
        "version": "2.0",
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("smart_pool_api:app", host="0.0.0.0", port=8000, reload=True)
