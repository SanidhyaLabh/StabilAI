import numpy as np
from database import get_all_sessions


def predict_next_psi(user_id="default", num_sessions=5):

    sessions = get_all_sessions(user_id)

    if len(sessions) < 3:
        return []

    psi = np.array([float(s["psi"]) for s in sessions])
    x = np.arange(len(psi))

    if len(sessions) > 1:
        slope, intercept = np.polyfit(x, psi, 1)
        
        future_x = np.arange(len(psi), len(psi) + num_sessions)
        predicted_psi = slope * future_x + intercept
        
        # Clamp between 0 and 100
        predicted_psi = np.clip(predicted_psi, 0, 100)
        
        return [round(p, 1) for p in predicted_psi]
    
    return []
