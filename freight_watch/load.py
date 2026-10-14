from pathlib import Path

import pandas as pd

from freight_watch import config

REQUIRED_COLUMNS = [
    "shipment_id",
    "origin",
    "destination",
    "route_type",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
    "shipment_date",
]


def week_start(dates: pd.Series) -> pd.Series:
    """Monday of the ISO week each date falls in (Sunday maps to the previous Monday)."""
    dates = pd.to_datetime(dates).dt.normalize()
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def load_shipments(path: Path = config.SHIPMENTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"shipments file is missing columns: {missing}")
    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("shipments file has blank values in required columns")
    if df["shipment_id"].duplicated().any():
        raise ValueError("shipments file has duplicate shipment_id values")
    for col in ("quantity_tonnes", "distance_km", "freight_cost_inr"):
        if (df[col] <= 0).any():
            raise ValueError(f"shipments file has non-positive values in {col}")

    df["shipment_date"] = pd.to_datetime(df["shipment_date"])
    df["route"] = df["origin"] + "-" + df["destination"]

    types_per_route = df.groupby("route")["route_type"].nunique()
    if (types_per_route > 1).any():
        bad = types_per_route[types_per_route > 1].index.tolist()
        raise ValueError(f"routes with more than one route_type: {bad}")

    df["week_of"] = week_start(df["shipment_date"])
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df
