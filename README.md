# ActiveBharat — Backend Service

> **Python FastAPI ML Engine** for real-time biomechanical analysis, pose estimation, Phase 2 kinematics (Mahalanobis distance, Quaternion torsion, Dimensionless Jerk), rPPG heart rate, and athlete intelligence.

---

## Architecture

```
activebharat-backend/
├── api_server.py         ← FastAPI application (main entrypoint)
├── pose_analyzer.py      ← MediaPipe wrapper + Phase 2 kinematics (math2.pdf)
├── rppg_processor.py     ← Remote PPG heart rate from face ROI
├── intelligence.py       ← Athlete performance AI + coaching engine
├── feature_extractor.py  ← ML feature extraction utilities
├── realtime_analyzer.py  ← Streaming analysis worker
├── generate_dataset.py   ← Olympic pose dataset generator
├── model_trainer.py      ← Train/export custom MLP models
├── requirements.txt
├── Dockerfile
└── .env.example
```

### Key Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/session/start` | Start a training session |
| `POST` | `/session/{id}/frame` | Analyse a JPEG frame (HTTP fallback) |
| `WS` | `/session/{id}/live-stream` | 60fps WebSocket stream (Phase 2 Edge AI) |
| `POST` | `/session/{id}/end` | End session and get summary |
| `GET` | `/leaderboard` | Global athlete rankings |
| `POST` | `/rppg/frame` | Heart rate from face image |
| `GET` | `/athlete/{id}` | Bio-passport data |

### Phase 2 Kinematics (math2.pdf)

The backend now implements the full Phase-Space kinematic suite:

- **Mahalanobis Distance** (`scipy.spatial.distance.mahalanobis`) — compares athlete's phase-space state vector `[θ, θ̇]` against a Gaussian ideal distribution
- **Quaternion Torsion** (`scipy.spatial.transform.Rotation`) — 3D torsional deviation from optimal joint rotation
- **Dimensionless Jerk** — Energy Efficiency Index from 3rd derivative of Centre of Mass trajectory
- **Sliding Window Buffer** — 60-frame `collections.deque` maintains state history for derivative calculations
- **HMM Phase State Machine** — Sparse evaluation of kinematics based on vertical CoM velocity

---

## Setup

### Local (Python virtualenv)

```bash
# 1. Clone and enter the repo
git clone https://github.com/your-org/activebharat-backend.git
cd activebharat-backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env

# 5. Create runtime directories
mkdir -p db models dataset

# 6. Start the server
python -m uvicorn api_server:app --host 0.0.0.0 --port 8082 --reload
```

### Docker

```bash
docker build -t activebharat-backend .
docker run -p 8082:8082 -v $(pwd)/db:/app/db -v $(pwd)/models:/app/models activebharat-backend
```

---

## Testing

```bash
# Health check
curl http://localhost:8082/health

# Start a session
curl -X POST http://localhost:8082/session/start \
  -H "Content-Type: application/json" \
  -d '{"sport": "vertical_jump", "athlete_id": "test_athlete"}'

# Run Phase 2 math unit tests
python test_math2_backend.py
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | REST API + WebSocket server |
| `mediapipe` | Pose landmark extraction (33 3D points) |
| `numpy` + `scipy` | Phase 2 matrix math (Mahalanobis, Quaternions) |
| `opencv-python` | Image decoding and preprocessing |
| `tensorflow` + `scikit-learn` | MLP model training and inference |
| `websockets` | Native WebSocket support |

---

## Connecting the Mobile App

The Android app (`activebharat-android`) must point to this server's IP:

```js
// In activebharat-android/src/constants.js
export const BACKEND_HOST = '192.168.x.x';  // Your Wi-Fi IP
export const BACKEND_PORT = 8083;            // Via frontend proxy
```

The frontend service (`activebharat-frontend`) proxies all `/api/*` and phone requests to this backend on port `8082`.
