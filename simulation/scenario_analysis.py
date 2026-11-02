"""
Multi-Scenario Comparative Analysis Engine.
Evaluates incremental intervention scenarios (10%, 20%, 30%, 40%, 50%)
for urban greening and generates structured comparative dataframes.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

import config
from simulation.mitigation_engine import simulate_mitigation
from models.regression_ann import LSTRegressionANN


def run_multi_scenario_analysis(
    mask: np.ndarray,
    metadata: Dict[str, float],
    regression_model: LSTRegressionANN,
    scaler: Optional[Any] = None,
    scenarios: Optional[List[float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates a suite of greening scenarios (default 10%, 20%, 30%, 40%, 50%).

    Returns:
        df: Pandas DataFrame comparing scenarios.
        scenario_results: Dictionary of full simulation results by scenario tag.
    """
    if scenarios is None:
        scenarios = config.DEFAULT_INTERVENTION_SCENARIOS

    records = []
    scenario_details = {}

    for frac in scenarios:
        res = simulate_mitigation(
            mask=mask,
            metadata=metadata,
            conversion_percentage=frac,
            regression_model=regression_model,
            scaler=scaler,
            contiguous=True
        )

        tag = f"{int(frac * 100)}%"
        scenario_details[tag] = res

        records.append({
            "Scenario": tag,
            "Greening Fraction": frac,
            "Original Impervious %": res["original_landcover_pcts"]["impervious_pct"],
            "Modified Impervious %": res["modified_landcover_pcts"]["impervious_pct"],
            "Original Vegetation %": res["original_landcover_pcts"]["vegetation_pct"],
            "Modified Vegetation %": res["modified_landcover_pcts"]["vegetation_pct"],
            "Original LST Anomaly (°C)": res["original_lst_anomaly"],
            "Modified LST Anomaly (°C)": res["modified_lst_anomaly"],
            "Predicted Cooling (°C)": res["predicted_temperature_reduction"],
            "Pixels Converted": res["pixels_converted"]
        })

    df = pd.DataFrame(records)
    return df, scenario_details
