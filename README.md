# VH11_RAILWAY

AI-Based Train Rescheduling System that includes:
Data Input (train details, delays, passenger load, weather, etc.)
Model Training (delay prediction + action classification with ML models like Random Forest)
Results Visualization (confusion matrix, delay prediction graphs, metrics)
Schedule Optimization (minimize delays, congestion, cancellations)
What-If Simulation (scenario testing, rerouting strategies, capacity management)

🚆 AI-Based Train Rescheduling System

An AI-powered system for train delay prediction, rescheduling, and optimization using machine learning and simulation. This project helps railway networks reduce congestion, minimize delays, and improve passenger satisfaction through predictive modeling, intelligent scheduling, and what-if scenario analysis.

✨ Features

📊 Data Input Module:
Train details (type, day, holiday, upstream delays, passenger load, weather, crew/platform availability).
Supports both real-time and batch input formats.

🤖 Machine Learning Models:
Delay Prediction (numeric delay in minutes).
Action Classification (NoChange / Delay / ShortTurn / Cancel).
Models supported: Random Forest, Decision Trees, etc.

📈 Model Training & Results:
Training accuracy & F1-score > 99%.
Metrics: MAE, RMSE, R² score for delay prediction.
Confusion matrix & classification reports.

🛠 Schedule Optimization:
AI-powered schedule re-planning with configurable objective weights (delay, cancellation, congestion).
Multiple optimization strategies (e.g., Greedy).

🔮 What-If Simulation:
Run scenario-based experiments for Normal Operations, Disruptions, and Peak Hours.
Evaluate rerouting aggressiveness, holding strategies, and capacity management.

⚙️ Installation
# Clone the repository
git clone https://github.com/your-username/AI-Train-Rescheduling-System.git
cd AI-Train-Rescheduling-System

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

1️⃣ Train Models
python src/training/train_models.py

2️⃣ Run Schedule Optimization
python src/optimization/run_optimizer.py

3️⃣ Launch Web App
streamlit run app/main.py

