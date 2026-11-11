"""scenarios.py stub for ri_drift"""
from typing import Any, Dict, List
import pandas as pd

IS_STUB = True

def inject_covariate_shift(df: pd.DataFrame, profile: Dict[str, Any], feature: str, magnitude: float, start_row: int) -> pd.DataFrame:
    df_out = df.copy()
    if feature in df_out.columns:
        df_out.iloc[start_row:, df_out.columns.get_loc(feature)] += magnitude
    return df_out

def inject_concept_shift(df: pd.DataFrame, profile: Dict[str, Any], region_rule: Any, flip_fraction: float, start_row: int) -> pd.DataFrame:
    return df.copy()

def make_invalid_upload(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df_out = df.copy()
    if kind == "drop_column" and len(df_out.columns) > 0:
        return df_out.drop(columns=[df_out.columns[0]])
    return df_out

def hold_back_labels(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    return df.copy()

def build_scenario_stream(batches: List[pd.DataFrame], scenario_name: str) -> List[pd.DataFrame]:
    return [b.copy() for b in batches]

