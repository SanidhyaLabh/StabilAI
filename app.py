from flask import Flask, render_template, request, redirect, jsonify
from tracker import start_tracking
from analysis import analyze_session
from database import init_db, save_session, get_all_sessions, get_last_session
from planner import generate_recommendation
from predictor import predict_next_psi
import json

app = Flask(__name__)

# Initialize DB
init_db()


# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")


# =========================
# TRAIN PAGE
# =========================
@app.route("/train")
def train():
    return render_template("train.html")


# =========================
# START TRAINING
# =========================
@app.route("/start", methods=["POST"])
def start():

    mode = request.form.get("mode")

    data = start_tracking(mode)
    analysis = analyze_session(data)

    save_session(
        "default",
        mode,
        data["psi"],
        data["tremor"],
        data["error"],
        data["depth_error"],
        data["pressure_dev"],
        data["trajectory"]
    )

    return render_template("result.html", data=data, analysis=analysis)


# =========================
# DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():

    sessions = get_all_sessions("default")

    labels = []
    psi = []
    tremor = []
    error = []
    depth = []

    for s in sessions:
        labels.append(s["timestamp"])
        psi.append(float(s["psi"]))
        tremor.append(float(s["tremor"]))
        error.append(float(s["error"]))
        depth.append(float(s["depth_error"]))

    recommendation = generate_recommendation("default")
    prediction = predict_next_psi("default", 5)

    return render_template(
        "dashboard.html",
        labels=labels,
        psi=psi,
        tremor=tremor,
        error=error,
        depth=depth,
        recommendation=recommendation,
        prediction=prediction,
        modes=labels # Fix for template using 'modes' or 'labels'
    )


# =========================
# REPORTS PAGE
# =========================
@app.route("/reports")
def reports():
    sessions = get_all_sessions("default")
    return render_template("reports.html", sessions=sessions)


# =========================
# HEATMAP / REPLAY
# =========================
@app.route("/replay")
def replay():
    return render_template("replay.html")

@app.route("/heatmap-data")
def heatmap_data():
    # Get the latest session for replay
    session = get_last_session("default")

    if not session:
        return jsonify([])

    # Trajectory is stored as a JSON string in DB, need to parse if it is, or if already list
    try:
        trajectory = json.loads(session["trajectory"])
    except:
        trajectory = [] # Fallback

    return jsonify(trajectory)


# =========================
# PLANNER / PROGRESS
# =========================
@app.route("/progress")
def progress():
    recommendation = generate_recommendation("default")
    prediction = predict_next_psi("default", 5)
    
    # Calculate simple stats for the view
    sessions = get_all_sessions("default")
    avg_psi = 0
    if sessions:
        avg_psi = sum([s["psi"] for s in sessions]) / len(sessions)

    return render_template(
        "progress.html", 
        recommendation=recommendation, 
        prediction=prediction,
        avg_psi=round(avg_psi, 1)
    )


# =========================
# LEADERBOARD
# =========================
@app.route("/leaderboard")
def leaderboard():
    # Mock leaderboard
    leaders = [
        {"name": "Dr. Strange", "score": 98.5},
        {"name": "House M.D.", "score": 96.2},
        {"name": "Meredith Grey", "score": 94.0},
        {"name": "You", "score": 0} # Placeholder
    ]
    
    # Try to get user max score
    sessions = get_all_sessions("default")
    if sessions:
        max_score = max([s["psi"] for s in sessions])
        leaders[3]["score"] = max_score

    leaders.sort(key=lambda x: x["score"], reverse=True)
    
    return render_template("leaderboard.html", leaders=leaders)


if __name__ == "__main__":
    app.run(debug=True)
