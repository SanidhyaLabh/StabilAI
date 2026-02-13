def analyze_session(data):

    feedback = []
    mistakes = []

    psi = data["psi"]

    # Tremor analysis
    if data["tremor"] > 5:
        feedback.append("Reduce hand tremor. Practice slow steady movements.")
        mistakes.append("High tremor detected")

    # Path deviation
    if data["error"] > 40:
        feedback.append("Improve trajectory accuracy. Follow guide path carefully.")
        mistakes.append("High path deviation")

    # Depth control
    if data["depth_error"] > 8:
        feedback.append("Maintain consistent penetration depth.")
        mistakes.append("Depth instability")

    # Pressure control
    if data["pressure_dev"] > 1:
        feedback.append("Apply uniform pressure while tracing.")
        mistakes.append("Pressure variation")

    if not feedback:
        feedback.append("Excellent surgical control. Maintain consistency.")

    return {
        "percentage": psi,
        "feedback": feedback,
        "mistakes": mistakes
    }
