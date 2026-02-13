import numpy as np
from database import get_all_sessions


def generate_recommendation(user_id="default"):

    sessions = get_all_sessions(user_id)

    if len(sessions) < 3:
        return {
            "recommended_mode": "line",
            "focus_metric": "Consistency",
            "goal": "Build baseline stability (5 sessions)",
            "trend": "Collecting Data"
        }

    tremor = np.array([float(s["tremor"]) for s in sessions])
    error = np.array([float(s["error"]) for s in sessions])
    depth = np.array([float(s["depth_error"]) for s in sessions])

    # Normalize variances to compare different metrics
    # (Simple normalization by mean to get coefficient of variation, or just raw variance if scales are similar)
    # Here we use raw variance but assume they are roughly comparable or weighted
    variances = {
        "tremor": np.var(tremor),
        "error": np.var(error),
        "depth": np.var(depth)
    }

    weakest = max(variances, key=variances.get)

    mapping = {
        "tremor": "micro",
        "error": "circle",
        "depth": "depth_drill"
    }

    # Determine trend (slope of PSI)
    psi = np.array([float(s["psi"]) for s in sessions])
    x = np.arange(len(psi))
    slope, _ = np.polyfit(x, psi, 1)
    
    trend_msg = "Steady Improvement" if slope > 0.5 else "Plateau Detected" if slope > -0.5 else "Declining Performance"

    return {
        "recommended_mode": mapping.get(weakest, "line"),
        "focus_metric": weakest.upper(),
        "goal": "Reduce variance in " + weakest,
        "trend": trend_msg
    }
