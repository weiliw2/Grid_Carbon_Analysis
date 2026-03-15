"""Shared scenario and calculator logic for the dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from assumptions import FOSSIL_FUELS, RENEWABLE_FUELS

COMPARISON_COUNTRIES = ["USA", "CHN", "IND", "DEU", "FRA", "GBR", "SGP", "NOR", "ISL"]


def get_country_row(country_data: pd.DataFrame, country_code: str) -> pd.Series:
    """Fetch a single country row from the aggregated country table."""
    return country_data[country_data["country"] == country_code].iloc[0]


def get_feature_value(features: pd.DataFrame, column: str) -> float:
    """Safely read a single feature value from a single-row DataFrame."""
    return float(features[column].iloc[0]) if column in features.columns else 0.0


def apply_coal_to_solar_shift(features: pd.DataFrame, coal_reduction: int) -> pd.DataFrame:
    """Update a country's feature row to reflect a coal-to-solar transition."""
    scenario_features = features.copy()
    if coal_reduction <= 0:
        return scenario_features

    if "Coal_pct" not in scenario_features.columns or "Solar_pct" not in scenario_features.columns:
        return scenario_features

    scenario_features["Coal_pct"] -= coal_reduction
    scenario_features["Solar_pct"] += coal_reduction

    renewable_cols = [f"{fuel}_pct" for fuel in RENEWABLE_FUELS if f"{fuel}_pct" in scenario_features.columns]
    fossil_cols = [f"{fuel}_pct" for fuel in FOSSIL_FUELS if f"{fuel}_pct" in scenario_features.columns]

    if renewable_cols:
        scenario_features["renewable_ratio"] = scenario_features[renewable_cols].sum(axis=1) / 100
    if fossil_cols:
        scenario_features["fossil_ratio"] = scenario_features[fossil_cols].sum(axis=1) / 100

    return scenario_features


def calculate_data_center_metrics(
    carbon_intensity: float,
    dc_power_mw: int,
    utilization: int,
    pue: float,
    carbon_tax: int,
) -> dict[str, float]:
    """Calculate annual energy, emissions, and carbon cost for a data center."""
    effective_power_mw = dc_power_mw * (utilization / 100) * pue
    annual_energy_gwh = effective_power_mw * 8760 / 1000
    annual_emissions_tonnes = (annual_energy_gwh * 1_000_000 * carbon_intensity) / 1_000_000_000
    annual_carbon_cost = annual_emissions_tonnes * carbon_tax

    return {
        "effective_power_mw": effective_power_mw,
        "annual_energy_gwh": annual_energy_gwh,
        "annual_emissions_tonnes": annual_emissions_tonnes,
        "annual_carbon_cost": annual_carbon_cost,
    }


def build_location_comparison(
    country_data: pd.DataFrame,
    annual_energy_gwh: float,
    carbon_tax: int,
    selected_country: str | None = None,
) -> pd.DataFrame:
    """Build a comparison table for a fixed annual electricity demand across countries."""
    comparison_data = []
    countries_to_compare = list(COMPARISON_COUNTRIES)

    if selected_country and selected_country not in countries_to_compare:
        countries_to_compare.append(selected_country)

    for country in countries_to_compare:
        if country not in country_data["country"].values:
            continue

        country_info = get_country_row(country_data, country)
        annual_emissions = (
            annual_energy_gwh * 1_000_000 * country_info["carbon_intensity_gco2_kwh"] / 1_000_000_000
        )
        comparison_data.append(
            {
                "Country": country,
                "Carbon Intensity": country_info["carbon_intensity_gco2_kwh"],
                "Annual Emissions (tonnes)": annual_emissions,
                "Annual Carbon Cost ($)": annual_emissions * carbon_tax,
                "Renewable %": country_info["renewable_percentage"],
            }
        )

    return pd.DataFrame(comparison_data).sort_values("Annual Emissions (tonnes)")
