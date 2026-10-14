import numpy as np
import pandas as pd

from freight_watch import config


def weekly_cost(shipments: pd.DataFrame) -> pd.DataFrame:
    """Cost per tonne-km for each (route, week): total cost / total tonne-km."""
    weekly = (
        shipments.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            freight_cost_inr=("freight_cost_inr", "sum"),
            tonne_km=("tonne_km", "sum"),
            shipments=("shipment_id", "count"),
        )
        .sort_values(["route", "week_of"], ignore_index=True)
    )
    weekly["cost_per_tonne_km"] = weekly["freight_cost_inr"] / weekly["tonne_km"]
    return weekly


def add_own_history(weekly: pd.DataFrame, window: int = config.HISTORY_WEEKS) -> pd.DataFrame:
    """Compare each week to the mean of up to `window` strictly-prior weeks on the same route.

    The first week of a route has no history, so its baseline and % are NaN.
    """
    weekly = weekly.sort_values(["route", "week_of"], ignore_index=True)
    weekly["own_baseline"] = weekly.groupby("route")["cost_per_tonne_km"].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )
    weekly["own_history_weeks"] = weekly.groupby("route").cumcount().clip(upper=window)
    weekly["own_pct"] = (weekly["cost_per_tonne_km"] / weekly["own_baseline"] - 1) * 100
    return weekly


def add_peer_comparison(weekly: pd.DataFrame) -> pd.DataFrame:
    """Compare each week to the simple mean of the other same-type routes in that week.

    Routes with no same-type peer in a week get NaN.
    """
    group = weekly.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    total = group.transform("sum")
    count = group.transform("count")
    peers = count - 1
    weekly["peer_count"] = peers
    weekly["peer_baseline"] = ((total - weekly["cost_per_tonne_km"]) / peers).where(peers > 0, np.nan)
    weekly["peer_pct"] = (weekly["cost_per_tonne_km"] / weekly["peer_baseline"] - 1) * 100
    return weekly


def compute_metrics(shipments: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly_cost(shipments)
    weekly = add_own_history(weekly)
    weekly = add_peer_comparison(weekly)
    return weekly.sort_values(["week_of", "route"], ignore_index=True)
