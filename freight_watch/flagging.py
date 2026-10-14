import pandas as pd

from freight_watch import config


def flag_candidates(
    weekly: pd.DataFrame,
    rising_min: float = config.OWN_RISING_MIN_PCT,
    own_abnormal: float = config.OWN_ABNORMAL_PCT,
    peer_abnormal: float = config.PEER_ABNORMAL_PCT,
) -> pd.DataFrame:
    """Mark route-weeks whose cost is rising and abnormal.

    Rising: own % > rising_min. Abnormal: own % >= own_abnormal or peer % >= peer_abnormal.
    Weeks with no own history (NaN own %) are never candidates; a NaN peer % only
    disables the peer check.
    """
    weekly = weekly.copy()
    own = weekly["own_pct"]
    peer = weekly["peer_pct"]

    weekly["is_rising"] = own.gt(rising_min)
    weekly["own_abnormal"] = own.ge(own_abnormal)
    weekly["peer_abnormal"] = peer.ge(peer_abnormal)
    weekly["is_candidate"] = weekly["is_rising"] & (weekly["own_abnormal"] | weekly["peer_abnormal"])
    return weekly


def candidates(weekly: pd.DataFrame, **thresholds: float) -> pd.DataFrame:
    """Only the candidate rows, sorted by week then route."""
    flagged = flag_candidates(weekly, **thresholds)
    return flagged[flagged["is_candidate"]].sort_values(["week_of", "route"], ignore_index=True)
