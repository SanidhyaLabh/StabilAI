import { useNavigate } from "react-router-dom";
import { useState } from "react";

export default function Train() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const startTraining = async (mode) => {
    setLoading(true);

    const res = await fetch("http://127.0.0.1:5000/start", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ mode }),
    });

    const data = await res.json();
    localStorage.setItem("sessionData", JSON.stringify(data));
    setLoading(false);
    navigate("/results");
  };

  return (
    <div className="train-page">
      <h1>Select Training Program</h1>

      <button onClick={() => startTraining("incision")}>
        Incision Training
      </button>

      <button onClick={() => startTraining("precision")}>
        Precision Training
      </button>

      {loading && <p>Training Running...</p>}
    </div>
  );
}
