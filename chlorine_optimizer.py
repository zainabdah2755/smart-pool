"""
Smart Pool - Chlorine Optimization Engine v2
=============================================
Multi-factor chlorine dosage calculator using:
  • pH-based demand curve
  • Temperature breakpoint dosing
  • Turbidity / bather-load estimate
  • Algae-risk pre-emptive shock
  • Time-of-day / UV degradation factor
  • Combined chlorine (chloramine) estimate
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ChlorineResult:
    dose_grams: int                    # recommended chlorine to add (g/m³)
    free_chlorine_target: float        # target free Cl ppm
    shock_required: bool               # breakpoint shock needed?
    rationale: list[str]               # human-readable explanation


def calculate_chlorine_dose(
    *,
    ph: float,
    temp_eau: float,
    ir: float,                         # turbidity 0-1
    humidite: float,
    luminosite: float,                 # LDR 0-1023
    algae_risk: float,                 # 0-100 from ML model
    pool_volume_m3: float = 50.0,      # default 50 m³ pool
    current_free_cl: Optional[float] = None,  # measured ppm if available
) -> ChlorineResult:
    """
    Returns an optimised chlorine dose in grams for the given conditions.
    All factor math is additive on top of a safe baseline,
    then scaled to pool volume.
    """
    rationale: list[str] = []

    # ── Baseline: maintain 1.0 ppm free chlorine ─────────────────
    base_ppm = 1.0
    rationale.append(f"Base target: {base_ppm} ppm free Cl")

    # ── pH factor: efficiency of chlorine drops above pH 7.5 ─────
    # At pH 7.0 → ~73 % active; pH 7.5 → ~48 %; pH 8.0 → ~22 %
    # We increase dose to compensate.
    if ph <= 7.5:
        ph_multiplier = 1.0
    elif ph <= 8.0:
        ph_multiplier = 1.4
        rationale.append(f"pH {ph:.2f} → +40 % dose (reduced Cl efficiency)")
    else:
        ph_multiplier = 1.8
        rationale.append(f"pH {ph:.2f} → +80 % dose (very low Cl efficiency)")

    # ── Temperature factor: higher temp = faster Cl degradation ──
    if temp_eau <= 25:
        temp_add_ppm = 0.0
    elif temp_eau <= 30:
        temp_add_ppm = 0.3
        rationale.append(f"Temp {temp_eau:.1f}°C → +0.3 ppm (moderate evaporation)")
    elif temp_eau <= 35:
        temp_add_ppm = 0.6
        rationale.append(f"Temp {temp_eau:.1f}°C → +0.6 ppm (high evaporation)")
    else:
        temp_add_ppm = 1.0
        rationale.append(f"Temp {temp_eau:.1f}°C → +1.0 ppm (extreme heat)")

    # ── UV degradation factor (sunlight destroys chlorine) ────────
    uv_index = luminosite / 1023 * 11
    if uv_index > 6:
        uv_add_ppm = round((uv_index - 6) * 0.05, 2)
        rationale.append(f"UV index ~{uv_index:.1f} → +{uv_add_ppm} ppm (UV degradation)")
    else:
        uv_add_ppm = 0.0

    # ── Turbidity / organic load factor ──────────────────────────
    if ir > 0.6:
        turbidity_add_ppm = 0.8
        rationale.append(f"High turbidity (IR={ir:.2f}) → +0.8 ppm (organic demand)")
    elif ir > 0.3:
        turbidity_add_ppm = 0.4
        rationale.append(f"Moderate turbidity (IR={ir:.2f}) → +0.4 ppm")
    else:
        turbidity_add_ppm = 0.0

    # ── Algae pre-treatment ───────────────────────────────────────
    if algae_risk >= 75:
        algae_add_ppm = 1.5
        rationale.append(f"Algae risk {algae_risk:.0f}/100 → +1.5 ppm (algaecide shock)")
    elif algae_risk >= 50:
        algae_add_ppm = 0.7
        rationale.append(f"Algae risk {algae_risk:.0f}/100 → +0.7 ppm (preventive)")
    else:
        algae_add_ppm = 0.0

    # ── Total target ppm ─────────────────────────────────────────
    total_ppm = (
        (base_ppm + temp_add_ppm + uv_add_ppm + turbidity_add_ppm + algae_add_ppm)
        * ph_multiplier
    )
    total_ppm = round(min(total_ppm, 5.0), 2)   # cap at 5 ppm (safe max)

    # Adjust if we know current free Cl
    if current_free_cl is not None and current_free_cl >= total_ppm:
        dose_grams = 0
        rationale.append(f"Current free Cl {current_free_cl} ppm ≥ target → no dose needed")
    else:
        deficit_ppm = total_ppm if current_free_cl is None else (total_ppm - current_free_cl)
        # 1 ppm in 1 m³ ≈ 1 gram of 100 % available Cl
        # Trichlor (common tablet) is ~90 % available Cl → ÷ 0.9
        dose_grams = int(math.ceil(deficit_ppm * pool_volume_m3 / 0.9))

    # ── Shock threshold ───────────────────────────────────────────
    shock_required = total_ppm > 3.0 or algae_risk >= 75 or ir > 0.7
    if shock_required:
        rationale.append("⚡ Breakpoint shock recommended (10× normal dose)")

    return ChlorineResult(
        dose_grams=dose_grams,
        free_chlorine_target=total_ppm,
        shock_required=shock_required,
        rationale=rationale,
    )


# ── Quick sanity test ─────────────────────────────────────────────
if __name__ == "__main__":
    result = calculate_chlorine_dose(
        ph=7.8, temp_eau=31, ir=0.45,
        humidite=65, luminosite=800,
        algae_risk=55, pool_volume_m3=50,
    )
    print(f"Dose: {result.dose_grams} g")
    print(f"Target: {result.free_chlorine_target} ppm")
    print(f"Shock: {result.shock_required}")
    for r in result.rationale:
        print(f"  • {r}")
