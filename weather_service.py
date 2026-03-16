"""
Smart Pool - Weather Integration Service
=========================================
Fetches current + 3-day forecast from Open-Meteo (free, no API key needed).
Derives pool-relevant weather impact scores.
"""

from __future__ import annotations
import httpx
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import math


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class CurrentWeather:
    temperature_2m: float           # °C
    relative_humidity_2m: float     # %
    precipitation: float            # mm
    wind_speed_10m: float           # km/h
    uv_index: float                 # 0-11
    weather_code: int               # WMO code
    weather_description: str

@dataclass
class WeatherForecastDay:
    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float        # mm
    uv_index_max: float
    weather_code: int

@dataclass
class WeatherImpact:
    """Pool-specific impact derived from weather data."""
    chlorine_extra_ppm: float       # extra Cl needed due to weather
    evaporation_level: str          # Low / Moderate / High
    contamination_risk: str         # Low / Moderate / High  (rain, wind)
    cover_recommendation: bool      # should pool be covered?
    maintenance_advisory: str       # human-readable advisory
    algae_weather_factor: float     # 0-1 multiplier for algae risk
    forecast_summary: list[str]     # 3-day bullet points

@dataclass
class WeatherData:
    current: Optional[CurrentWeather] = None
    forecast: list[WeatherForecastDay] = field(default_factory=list)
    impact: Optional[WeatherImpact] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────
# WMO CODE → DESCRIPTION
# ─────────────────────────────────────────────

WMO_DESCRIPTIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ hail", 99: "Thunderstorm w/ heavy hail",
}


# ─────────────────────────────────────────────
# FETCH WEATHER
# ─────────────────────────────────────────────

async def fetch_weather(lat: float, lon: float) -> WeatherData:
    """
    Fetches current weather + 3-day forecast from Open-Meteo.
    Falls back gracefully if the request fails.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "wind_speed_10m", "uv_index", "weather_code",
        ],
        "daily": [
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "uv_index_max", "weather_code",
        ],
        "forecast_days": 4,
        "timezone": "auto",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()
    except Exception as exc:
        return WeatherData(error=str(exc))

    # ── Parse current ─────────────────────────────────────────────
    c = raw.get("current", {})
    current = CurrentWeather(
        temperature_2m=c.get("temperature_2m", 0),
        relative_humidity_2m=c.get("relative_humidity_2m", 0),
        precipitation=c.get("precipitation", 0),
        wind_speed_10m=c.get("wind_speed_10m", 0),
        uv_index=c.get("uv_index", 0),
        weather_code=c.get("weather_code", 0),
        weather_description=WMO_DESCRIPTIONS.get(c.get("weather_code", 0), "Unknown"),
    )

    # ── Parse 3-day forecast ──────────────────────────────────────
    d = raw.get("daily", {})
    dates     = d.get("time", [])
    temp_max  = d.get("temperature_2m_max", [])
    temp_min  = d.get("temperature_2m_min", [])
    precip    = d.get("precipitation_sum", [])
    uv_max    = d.get("uv_index_max", [])
    w_codes   = d.get("weather_code", [])

    forecast = []
    for i in range(1, min(4, len(dates))):   # skip today (index 0), next 3 days
        forecast.append(WeatherForecastDay(
            date=dates[i],
            temp_max=temp_max[i] if i < len(temp_max) else 0,
            temp_min=temp_min[i] if i < len(temp_min) else 0,
            precipitation_sum=precip[i] if i < len(precip) else 0,
            uv_index_max=uv_max[i] if i < len(uv_max) else 0,
            weather_code=w_codes[i] if i < len(w_codes) else 0,
        ))

    # ── Compute pool impact ───────────────────────────────────────
    impact = _compute_impact(current, forecast)

    return WeatherData(current=current, forecast=forecast, impact=impact)


# ─────────────────────────────────────────────
# POOL IMPACT CALCULATION
# ─────────────────────────────────────────────

def _compute_impact(
    current: CurrentWeather,
    forecast: list[WeatherForecastDay],
) -> WeatherImpact:
    advisory_parts: list[str] = []
    forecast_summary: list[str] = []
    extra_cl = 0.0
    algae_factor = 0.0

    # ── UV impact on chlorine ─────────────────────────────────────
    if current.uv_index > 8:
        extra_cl += 0.5
        advisory_parts.append("Extreme UV: chlorine degrades faster — dose in the evening")
    elif current.uv_index > 5:
        extra_cl += 0.2

    # ── Rain dilution / contamination ────────────────────────────
    if current.precipitation > 10:
        extra_cl += 0.6
        advisory_parts.append("Heavy rain: organic contaminants entering pool — shock recommended")
        contamination_risk = "High"
    elif current.precipitation > 2:
        extra_cl += 0.3
        advisory_parts.append("Rain detected: check and adjust chemical balance")
        contamination_risk = "Moderate"
    else:
        contamination_risk = "Low"

    # ── Wind / debris ─────────────────────────────────────────────
    if current.wind_speed_10m > 30:
        extra_cl += 0.2
        advisory_parts.append("Strong wind: debris contamination — clean skimmer basket")

    # ── Evaporation ───────────────────────────────────────────────
    heat_load = current.temperature_2m + current.relative_humidity_2m * 0.1
    if heat_load > 45:
        evaporation_level = "High"
        extra_cl += 0.3
        advisory_parts.append("High evaporation: check water level daily")
    elif heat_load > 35:
        evaporation_level = "Moderate"
    else:
        evaporation_level = "Low"

    # ── Algae weather factor ──────────────────────────────────────
    # Warm + humid + sunny = high algae weather risk
    algae_factor = min(1.0, (
        max(0, (current.temperature_2m - 22) / 20)
        + max(0, (current.relative_humidity_2m - 60) / 80)
        + max(0, (current.uv_index - 4) / 10)
    ) / 3)

    if algae_factor > 0.6:
        advisory_parts.append("Weather conditions favor algae growth — maintain Cl ≥ 2 ppm")

    # ── Cover recommendation ──────────────────────────────────────
    cover_recommendation = (
        current.weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99]
        or current.wind_speed_10m > 35
    )

    # ── 3-day forecast summary ────────────────────────────────────
    for day in forecast:
        desc = WMO_DESCRIPTIONS.get(day.weather_code, "Unknown")
        line = (
            f"{day.date}: {desc}, "
            f"{day.temp_min:.0f}–{day.temp_max:.0f}°C, "
            f"UV {day.uv_index_max:.0f}"
        )
        if day.precipitation_sum > 5:
            line += f", 🌧 {day.precipitation_sum:.0f} mm rain"
        forecast_summary.append(line)

    if not advisory_parts:
        advisory_parts.append("Weather conditions are favourable for pool use")

    return WeatherImpact(
        chlorine_extra_ppm=round(extra_cl, 2),
        evaporation_level=evaporation_level,
        contamination_risk=contamination_risk,
        cover_recommendation=cover_recommendation,
        maintenance_advisory=" | ".join(advisory_parts),
        algae_weather_factor=round(algae_factor, 3),
        forecast_summary=forecast_summary,
    )


# ─────────────────────────────────────────────
# SYNC WRAPPER (for FastAPI sync routes)
# ─────────────────────────────────────────────

def fetch_weather_sync(lat: float, lon: float) -> WeatherData:
    """Synchronous wrapper — runs the async function in a new event loop."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, fetch_weather(lat, lon))
                return future.result()
        return loop.run_until_complete(fetch_weather(lat, lon))
    except Exception as exc:
        return WeatherData(error=str(exc))


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Marrakesh coordinates
    data = fetch_weather_sync(31.6295, -7.9811)
    if data.error:
        print("Weather fetch error:", data.error)
    else:
        print("Current:", data.current)
        print("Impact:", data.impact)
        for day in data.forecast:
            print(" Forecast:", day)
