"""
=============================================================================
ActiveBharat — FastAPI REST Server (Production-Fixed)
=============================================================================
Fixed in this version:
  - Pydantic v2: frame.dict() → frame.model_dump()
  - FastAPI lifespan pattern (replaces deprecated @on_event)
  - CORS: removed allow_credentials=True when allow_origins=["*"]
  - WebSocket: safe removal from connections list
  - _compute_xp moved to module level
  - Proper error handling throughout

Base URL: http://localhost:8000
Docs:     http://localhost:8000/docs

Start:
  cd ml/
  python api_server.py
  # OR: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
=============================================================================
"""

import json, csv, uuid, os, time, asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("[ERROR] FastAPI not installed. Run: pip install fastapi uvicorn")


# ─── Storage ──────────────────────────────────────────────────────────────────

DB_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "db"
DB_PATH.mkdir(parents=True, exist_ok=True)

DATASET_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "dataset"

SESSION_DB: Dict[str, dict] = {}
ATHLETE_DB: Dict[str, dict] = {}
FRAME_BUFFER: Dict[str, List[dict]] = defaultdict(list)
WS_CONNECTIONS: Dict[str, List[WebSocket]] = defaultdict(list)

# ─── Async Analysis Queue (decouples frame upload from MediaPipe processing) ───
# Phone posts a frame → stored instantly, returns 202 → worker processes async.
# Results stored in RESULT_STORE; phone polls /latest-result; dashboard gets WS push.
ANALYSIS_QUEUE: asyncio.Queue = None          # initialised in lifespan
RESULT_STORE: Dict[str, dict] = {}            # session_id → latest analysis result
_POSE_ANALYZERS: Dict[str, object] = {}       # session_id → PoseAnalyzer instance
RPPG_STORE: Dict[str, object] = {}            # session_id → RPPGProcessor instance


def _load_db():
    sessions_file = DB_PATH / "sessions.json"
    athletes_file = DB_PATH / "athletes.json"
    if sessions_file.exists():
        try:
            with open(sessions_file, encoding="utf-8") as f:
                SESSION_DB.update(json.load(f))
        except Exception as e:
            print(f"[DB WARN] Could not load sessions: {e}")
    if athletes_file.exists():
        try:
            with open(athletes_file, encoding="utf-8") as f:
                ATHLETE_DB.update(json.load(f))
        except Exception as e:
            print(f"[DB WARN] Could not load athletes: {e}")
    # Seed default athletes if empty
    if not ATHLETE_DB:
        ATHLETE_DB.update({
            "athlete_01": {"id": "athlete_01", "name": "Viraj Sharma",  "sport": "vertical_jump", "tier": "District", "bpi": 12450, "sessions": 0, "avatar": "VS"},
            "athlete_02": {"id": "athlete_02", "name": "Priya Desai",   "sport": "sprint",         "tier": "State",    "bpi": 11800, "sessions": 0, "avatar": "PD"},
            "athlete_03": {"id": "athlete_03", "name": "Rajan Mehta",   "sport": "snatch",         "tier": "National", "bpi": 14200, "sessions": 0, "avatar": "RM"},
            "athlete_04": {"id": "athlete_04", "name": "Amita Joshi",   "sport": "javelin",        "tier": "District", "bpi": 9300,  "sessions": 0, "avatar": "AJ"},
            "athlete_05": {"id": "athlete_05", "name": "Karan Singh",   "sport": "cricket_bat",    "tier": "Block",    "bpi": 8600,  "sessions": 0, "avatar": "KS"},
        })
    print(f"[DB] {len(SESSION_DB)} sessions, {len(ATHLETE_DB)} athletes loaded")


def _save_db():
    try:
        with open(DB_PATH / "sessions.json", "w", encoding="utf-8") as f:
            json.dump(SESSION_DB, f, indent=2, default=str)
        with open(DB_PATH / "athletes.json", "w", encoding="utf-8") as f:
            json.dump(ATHLETE_DB, f, indent=2)
    except Exception as e:
        print(f"[DB WARN] Could not save db: {e}")


# ─── XP Computation (module-level, not inside FASTAPI_AVAILABLE block) ────────

def _compute_xp(scores: List[float], jump_heights: List[float]) -> int:
    base = 50
    if scores:
        avg = sum(scores) / len(scores)
        base += 200 if avg >= 90 else 120 if avg >= 75 else 60 if avg >= 55 else 20
    if jump_heights:
        base += min(int(max(jump_heights) * 2), 100)
    return base


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    athlete_id: str = "athlete_01"
    sport: str = "vertical_jump"

    model_config = {"json_schema_extra": {"example": {"athlete_id": "athlete_01", "sport": "vertical_jump"}}}


class FrameData(BaseModel):
    hip_angle_l: float = 0.0
    hip_angle_r: float = 0.0
    knee_angle_l: float = 0.0
    knee_angle_r: float = 0.0
    shoulder_angle_l: float = 0.0
    shoulder_angle_r: float = 0.0
    elbow_angle_l: float = 0.0
    elbow_angle_r: float = 0.0
    ankle_dorsiflexion_l: float = 0.0
    ankle_dorsiflexion_r: float = 0.0
    trunk_lean: float = 0.0
    spine_deviation: float = 0.0
    shoulder_hip_sep: float = 0.0
    head_forward_pos: float = 0.0
    com_height_norm: float = 0.5
    estimated_jump_height: float = 0.0
    limb_symmetry_idx: float = 1.0
    form_score: float = 0.0
    form_quality: str = "unknown"
    primary_feedback: str = ""
    phase: str = "setup"
    # Real camera frame from phone (base64 JPEG, optional)
    # When present, the server runs real MediaPipe analysis and overwrites the metrics above
    image_b64: Optional[str] = None


class NewAthleteRequest(BaseModel):
    name: str
    sport: str = "vertical_jump"
    tier: str = "Block"


# ─── FastAPI App with Lifespan ────────────────────────────────────────────────

if FASTAPI_AVAILABLE:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global ANALYSIS_QUEUE
        # Startup
        _load_db()
        ANALYSIS_QUEUE = asyncio.Queue(maxsize=50)   # drop frames if queue full
        task = asyncio.create_task(analysis_worker())
        print("[API] ActiveBharat API running → http://localhost:8082")
        print("[API] Swagger docs → http://localhost:8082/docs")
        print("[API] Async analysis worker started")
        yield
        # Shutdown
        task.cancel()
        _save_db()
        print("[API] DB saved. Shutting down.")

    async def _broadcast(session_id: str, payload: dict):
        """Send JSON payload to all WebSocket clients watching session_id."""
        dead = []
        text = json.dumps(payload, default=str)
        for ws in list(WS_CONNECTIONS.get(session_id, [])):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns = WS_CONNECTIONS.get(session_id, [])
            if ws in conns:
                conns.remove(ws)

    async def analysis_worker():
        """
        Background loop — runs MediaPipe on queued frames without blocking HTTP.
        Puts results in RESULT_STORE and broadcasts to dashboard WebSocket.
        """
        while True:
            try:
                item = await ANALYSIS_QUEUE.get()
                session_id, image_b64, sport, frame_dict = item
                try:
                    from pose_analyzer import PoseAnalyzer
                    if session_id not in _POSE_ANALYZERS:
                        _POSE_ANALYZERS[session_id] = PoseAnalyzer(sport=sport)
                    analyzer = _POSE_ANALYZERS[session_id]
                    result = analyzer.analyze_base64_image(image_b64, sport)

                    if result.get("pose_detected"):
                        angles = result.get("joint_angles", {})
                        update = {
                            "form_score":        result["form_score"],
                            "form_quality":      result["form_quality"],
                            "primary_feedback":  result["primary_feedback"],
                            "phase":             result["phase"],
                            "limb_symmetry_idx": result["symmetry_score"],
                            "trunk_lean":        result["trunk_lean"],
                            "estimated_jump_height": result.get("estimated_jump_height", 0.0),
                            "knee_angle_l":  angles.get("KNEE_L", 0.0),
                            "knee_angle_r":  angles.get("KNEE_R", 0.0),
                            "hip_angle_l":   angles.get("HIP_L", 0.0),
                            "hip_angle_r":   angles.get("HIP_R", 0.0),
                            "elbow_angle_l": angles.get("ELBOW_L", 0.0),
                            "elbow_angle_r": angles.get("ELBOW_R", 0.0),
                            "pose_detected": True,
                        }
                        frame_dict.update(update)

                        result_entry = {
                            **update,
                            "frame_num":    frame_dict["frame_num"],
                            "sport":        sport,
                            "analyzed_at":  time.time(),
                            "data_source":  "real",
                            # keypoints for ghost skeleton (normalized 0-1)
                            "keypoints":    result.get("keypoints", []),
                        }
                        RESULT_STORE[session_id] = result_entry

                        # Update the stored frame with real values
                        if FRAME_BUFFER.get(session_id):
                            FRAME_BUFFER[session_id][-1].update(update)

                        print(f"[AI] session={session_id[:8]} score={result['form_score']:.0f} "
                              f"quality={result['form_quality']} phase={result['phase']}")

                        # Broadcast to dashboard
                        await _broadcast(session_id, {"type": "frame", **result_entry})
                    else:
                        print(f"[AI] No pose in frame (session {session_id[:8]})")
                        await _broadcast(session_id, {
                            "type": "frame",
                            "frame_num":   frame_dict["frame_num"],
                            "pose_detected": False,
                            "analyzed_at": time.time(),
                        })

                except Exception as e:
                    print(f"[WORKER ERROR] {e}")
                finally:
                    ANALYSIS_QUEUE.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[WORKER FATAL] {e}")
                await asyncio.sleep(1)

    app = FastAPI(
        title="ActiveBharat API",
        description="Sports Biomechanics REST API powering the Android app and dashboard",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS: allow_credentials cannot be True with wildcard origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,  # FIX: must be False when allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────────

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "service": "ActiveBharat Sports Analysis API",
            "version": "2.0.0",
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "endpoints": {
                "sessions": "/sessions",
                "start_session": "POST /session/start",
                "add_frame": "POST /session/{id}/frame",
                "end_session": "POST /session/{id}/end",
                "athletes": "/athletes",
                "leaderboard": "/leaderboard",
                "live_stream": "ws://HOST:8000/metrics/live/{session_id}",
                "dataset": "/dataset/export",
                "docs": "/docs"
            }
        }

    @app.get("/health", tags=["Health"])
    async def health():
        return {
            "status": "ok",
            "sessions_count": len(SESSION_DB),
            "athletes_count": len(ATHLETE_DB),
            "dataset_ready": (DATASET_PATH / "training_data.csv").exists(),
            "model_ready": (Path(os.path.dirname(os.path.abspath(__file__))) / "models" / "pose_classifier.tflite").exists(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    # ── Sessions ──────────────────────────────────────────────────────────────

    @app.post("/session/start", tags=["Sessions"])
    async def start_session(req: StartSessionRequest):
        """Start a new biomechanical analysis session."""
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "athlete_id": req.athlete_id,
            "sport": req.sport,
            "status": "active",
            "started_at": datetime.utcnow().isoformat() + "Z",
            "ended_at": None,
            "frame_count": 0,
            "summary": None,
        }
        SESSION_DB[session_id] = session
        FRAME_BUFFER[session_id] = []
        return {"session_id": session_id, "sport": req.sport, "athlete_id": req.athlete_id, "message": "Session started"}

    @app.post("/session/{session_id}/frame", tags=["Sessions"])
    async def add_frame(session_id: str, frame: FrameData):
        """
        Fast frame ingestion — returns 202 immediately.
        If 'image_b64' provided: queued for async MediaPipe analysis.
        Poll GET /session/{id}/latest-result to get the analysis output.
        """
        if session_id not in SESSION_DB:
            raise HTTPException(404, "Session not found")
        if SESSION_DB[session_id]["status"] != "active":
            raise HTTPException(400, "Session not active")

        sport = SESSION_DB[session_id].get("sport", "vertical_jump")

        # Store frame immediately (without image to save memory)
        frame_dict = frame.model_dump()
        image_b64  = frame_dict.pop("image_b64", None)   # strip before DB
        frame_dict["frame_num"] = len(FRAME_BUFFER[session_id])
        frame_dict["timestamp"] = time.time()
        frame_dict["pose_detected"] = None  # will be updated by worker
        FRAME_BUFFER[session_id].append(frame_dict)
        SESSION_DB[session_id]["frame_count"] += 1

        # Queue for async MediaPipe analysis (non-blocking)
        if image_b64 and ANALYSIS_QUEUE is not None:
            try:
                ANALYSIS_QUEUE.put_nowait((session_id, image_b64, sport, frame_dict))
            except asyncio.QueueFull:
                print(f"[WARN] Analysis queue full, dropping frame for {session_id[:8]}")

        # Return immediately — phone polls /latest-result for analysis output
        latest = RESULT_STORE.get(session_id, {})
        return {
            "frame_num":        frame_dict["frame_num"],
            "status":           "queued" if image_b64 else "stored",
            "queued_for_analysis": bool(image_b64),
            # Include last known result so phone gets something on first call
            "last_form_score":  latest.get("form_score"),
            "last_feedback":    latest.get("primary_feedback"),
            "last_quality":     latest.get("form_quality"),
        }

    @app.get("/session/{session_id}/latest-result", tags=["Sessions"])
    async def latest_result(session_id: str):
        """
        Phone polls this endpoint every 3-4s to get the latest AI analysis result
        without waiting for the MediaPipe processing to complete.
        """
        if session_id not in SESSION_DB:
            raise HTTPException(404, "Session not found")
        result = RESULT_STORE.get(session_id)
        if not result:
            return {
                "session_id":  session_id,
                "data_source": "none",
                "message":     "No analysis yet — send frames with image_b64",
            }
        return {"session_id": session_id, **result}

    @app.post("/session/calibrate", tags=["Sessions"])
    async def calibrate_pose(frame: FrameData):
        """
        Lightweight calibration endpoint for Ghost Skeleton screen.
        Does NOT create a session or write to DB — just runs MediaPipe and
        returns keypoints + deviations for real-time skeleton overlay.
        """
        if not frame.image_b64:
            raise HTTPException(400, "image_b64 required for calibration")

        sport = "vertical_jump"   # default; caller can override via query param
        try:
            from pose_analyzer import PoseAnalyzer
            analyzer = PoseAnalyzer(sport=sport)
            result = analyzer.analyze_base64_image(frame.image_b64, sport)

            if not result.get("pose_detected"):
                return {
                    "pose_detected": False,
                    "form_score":    0,
                    "keypoints":     [],
                    "deviations":    {},
                    "primary_feedback": "Move into frame — stand 1.5–2m from camera",
                }

            angles  = result.get("joint_angles", {})
            kp      = result.get("keypoints", [])

            # Compute per-joint deviation from ideal (simple threshold)
            IDEAL = {
                "KNEE_L": 170, "KNEE_R": 170,
                "HIP_L":  170, "HIP_R":  170,
                "ELBOW_L": 160, "ELBOW_R": 160,
            }
            deviations = {}
            for joint, ideal_angle in IDEAL.items():
                actual = angles.get(joint, ideal_angle)
                diff   = abs(actual - ideal_angle)
                deviations[joint] = "good" if diff < 15 else "warning" if diff < 35 else "critical"

            return {
                "pose_detected":    True,
                "form_score":       result["form_score"],
                "form_quality":     result["form_quality"],
                "primary_feedback": result["primary_feedback"],
                "keypoints":        kp,
                "joint_angles":     angles,
                "deviations":       deviations,
                "symmetry_score":   result["symmetry_score"],
            }
        except Exception as e:
            print(f"[CALIBRATE ERROR] {e}")
            return {
                "pose_detected": False,
                "form_score":    0,
                "keypoints":     [],
                "deviations":    {},
                "primary_feedback": "Calibration unavailable — check server logs",
            }

    @app.post("/session/{session_id}/end", tags=["Sessions"])
    async def end_session(session_id: str):
        """End a session and compute aggregate summary statistics."""
        if session_id not in SESSION_DB:
            raise HTTPException(404, "Session not found")
        if SESSION_DB[session_id]["status"] != "active":
            raise HTTPException(400, "Session already ended")

        frames = FRAME_BUFFER.get(session_id, [])
        if not frames:
            # Return minimal summary even if no frames (don't crash)
            summary = {"session_id": session_id, "athlete_id": SESSION_DB[session_id]["athlete_id"],
                       "sport": SESSION_DB[session_id]["sport"], "total_frames": 0, "valid_frames": 0,
                       "avg_form_score": 0, "peak_form_score": 0, "peak_jump_height_cm": 0,
                       "avg_jump_height_cm": 0, "avg_symmetry": 0, "xp_earned": 50,
                       "quality_distribution": {"elite": 0, "good": 0, "average": 0, "poor": 0}}
        else:
            valid = [f for f in frames if f.get("form_score", 0) > 0]
            scores = [f["form_score"] for f in valid]
            jump_heights = [f["estimated_jump_height"] for f in frames if f.get("estimated_jump_height", 0) > 5]
            symmetries = [f["limb_symmetry_idx"] for f in valid]
            quality_counts = {"elite": 0, "good": 0, "average": 0, "poor": 0}
            for f in frames:
                q = f.get("form_quality", "unknown")
                if q in quality_counts:
                    quality_counts[q] += 1

            duration = frames[-1]["timestamp"] - frames[0]["timestamp"] if len(frames) > 1 else 0
            summary = {
                "session_id": session_id,
                "athlete_id": SESSION_DB[session_id]["athlete_id"],
                "sport": SESSION_DB[session_id]["sport"],
                "total_frames": len(frames),
                "valid_frames": len(valid),
                "duration_seconds": round(duration, 1),
                "avg_form_score": round(sum(scores) / max(len(scores), 1), 1),
                "peak_form_score": round(max(scores, default=0), 1),
                "peak_jump_height_cm": round(max(jump_heights, default=0), 1),
                "avg_jump_height_cm": round(sum(jump_heights) / max(len(jump_heights), 1), 1),
                "avg_symmetry": round(sum(symmetries) / max(len(symmetries), 1), 3),
                "quality_distribution": quality_counts,
                "xp_earned": _compute_xp(scores, jump_heights),
            }

        SESSION_DB[session_id]["status"] = "completed"
        SESSION_DB[session_id]["ended_at"] = datetime.utcnow().isoformat() + "Z"
        SESSION_DB[session_id]["summary"] = summary

        # Update athlete BPI
        athlete_id = SESSION_DB[session_id]["athlete_id"]
        if athlete_id in ATHLETE_DB:
            ATHLETE_DB[athlete_id]["sessions"] = ATHLETE_DB[athlete_id].get("sessions", 0) + 1
            ATHLETE_DB[athlete_id]["bpi"] = ATHLETE_DB[athlete_id].get("bpi", 0) + summary["xp_earned"]

        _save_db()
        return summary

    @app.get("/session/{session_id}", tags=["Sessions"])
    async def get_session(session_id: str):
        if session_id not in SESSION_DB:
            raise HTTPException(404, "Session not found")
        return SESSION_DB[session_id]

    @app.get("/sessions", tags=["Sessions"])
    async def list_sessions(
        athlete_id: Optional[str] = None,
        sport: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = Query(default=20, le=100),
        offset: int = 0,
    ):
        sessions = list(SESSION_DB.values())
        if athlete_id:
            sessions = [s for s in sessions if s.get("athlete_id") == athlete_id]
        if sport:
            sessions = [s for s in sessions if s.get("sport") == sport]
        if status:
            sessions = [s for s in sessions if s.get("status") == status]
        sessions.sort(key=lambda x: x.get("started_at", ""), reverse=True)
        return {"total": len(sessions), "offset": offset, "limit": limit, "sessions": sessions[offset:offset + limit]}

    # ── Athletes ──────────────────────────────────────────────────────────────

    @app.get("/athletes", tags=["Athletes"])
    async def list_athletes():
        athletes = list(ATHLETE_DB.values())
        athletes.sort(key=lambda x: x.get("bpi", 0), reverse=True)
        return {"athletes": athletes, "total": len(athletes)}

    @app.post("/athlete", tags=["Athletes"])
    async def create_athlete(req: NewAthleteRequest):
        athlete_id = f"athlete_{len(ATHLETE_DB) + 1:03d}"
        initials = "".join(w[0].upper() for w in req.name.strip().split()[:2])
        athlete = {
            "id": athlete_id, "name": req.name, "sport": req.sport,
            "tier": req.tier, "bpi": 1000, "sessions": 0,
            "avatar": initials, "created_at": datetime.utcnow().isoformat() + "Z",
        }
        ATHLETE_DB[athlete_id] = athlete
        _save_db()
        return athlete

    @app.get("/athlete/{athlete_id}", tags=["Athletes"])
    async def get_athlete(athlete_id: str):
        if athlete_id not in ATHLETE_DB:
            raise HTTPException(404, "Athlete not found")
        athlete = dict(ATHLETE_DB[athlete_id])
        athlete["recent_sessions"] = sorted(
            [s for s in SESSION_DB.values()
             if s.get("athlete_id") == athlete_id and s.get("status") == "completed"],
            key=lambda x: x.get("started_at", ""), reverse=True
        )[:10]
        return athlete

    @app.get("/athlete/{athlete_id}/insights", tags=["Intelligence"])
    async def athlete_insights(athlete_id: str):
        """AI trend, anomalies, coaching drills, tier prediction."""
        if athlete_id not in ATHLETE_DB:
            raise HTTPException(404, "Athlete not found")
        try:
            from intelligence import generate_insights
        except ImportError:
            return {"error": "intelligence.py not found", "athlete_id": athlete_id}
        athlete  = ATHLETE_DB[athlete_id]
        sessions = sorted(
            [s for s in SESSION_DB.values()
             if s.get("athlete_id") == athlete_id and s.get("status") == "completed"],
            key=lambda x: x.get("started_at", "")
        )
        return generate_insights(athlete, sessions)

    @app.get("/session/{session_id}/coaching", tags=["Intelligence"])
    async def session_coaching(session_id: str):
        """Per-session AI coaching with targeted drills."""
        if session_id not in SESSION_DB:
            raise HTTPException(404, "Session not found")
        session = SESSION_DB[session_id]
        frames  = FRAME_BUFFER.get(session_id, [])
        try:
            from intelligence import CoachRecommender
        except ImportError:
            return {"error": "intelligence.py not found"}
        sport     = session.get("sport", "vertical_jump")
        summary   = session.get("summary", {})
        avg_score = summary.get("avg_form_score", 0) if summary else 0
        trend_g   = "improving" if avg_score >= 75 else "stable" if avg_score >= 55 else "declining"
        joint_keys = ["knee_angle_l", "knee_angle_r", "hip_angle_l", "hip_angle_r"]
        avg_angles: dict = {}
        if frames:
            for key in joint_keys:
                vals = [f.get(key, 0) for f in frames if f.get(key, 0) > 0]
                if vals:
                    avg_angles[key.replace("angle_", "").upper()] = sum(vals) / len(vals)
        coach = CoachRecommender().recommend(sport, avg_angles, trend_g)
        return {"session_id": session_id, "sport": sport, "avg_score": round(avg_score, 1), "coaching": coach}

    # ── Leaderboard ───────────────────────────────────────────────────────────

    @app.get("/leaderboard", tags=["Leaderboard"])
    async def get_leaderboard(
        sport: Optional[str] = None,
        limit: int = Query(default=20, le=50)
    ):
        athletes = list(ATHLETE_DB.values())
        if sport:
            athletes = [a for a in athletes if a.get("sport") == sport]
        athletes.sort(key=lambda x: x.get("bpi", 0), reverse=True)
        ranked = [{"rank": i + 1, **a} for i, a in enumerate(athletes[:limit])]
        return {"leaderboard": ranked, "sport": sport or "all", "total": len(athletes)}

    # ── Active Sessions (for dashboard auto-connect) ──────────────────────────

    @app.get("/sessions/active", tags=["Sessions"])
    async def get_active_sessions():
        """Returns all currently running sessions. Dashboard polls this to auto-connect."""
        active = [
            {
                "session_id": sid,
                "athlete_id": s.get("athlete_id"),
                "sport":      s.get("sport"),
                "started_at": s.get("started_at"),
                "frame_count": len(FRAME_BUFFER.get(sid, [])),
                "latest_score": RESULT_STORE.get(sid, {}).get("form_score"),
            }
            for sid, s in SESSION_DB.items()
            if s.get("status") == "active"
        ]
        return {"active_sessions": active, "count": len(active)}

    # ── rPPG Heart Rate Endpoints ─────────────────────────────────────────────

    class RPPGFrameRequest(BaseModel):
        session_id: str
        image_b64:  str
        timestamp:  Optional[float] = None

    @app.post("/rppg/frame", tags=["rPPG"])
    async def rppg_add_frame(req: RPPGFrameRequest):
        """Accept one camera frame for rPPG heart rate extraction."""
        try:
            from rppg_processor import RPPGProcessor
        except ImportError:
            return {"error": "rppg_processor.py not found"}

        sid = req.session_id
        if sid not in RPPG_STORE:
            RPPG_STORE[sid] = RPPGProcessor()

        proc: RPPGProcessor = RPPG_STORE[sid]
        frame_result = proc.add_frame(req.image_b64, req.timestamp)

        # Always compute — proc.compute() returns cached 'warmup' state fast when < MIN_FRAMES
        result = proc.compute()

        return {"frame": frame_result, "result": result}

    @app.get("/rppg/result/{session_id}", tags=["rPPG"])
    async def rppg_get_result(session_id: str):
        """Get latest rPPG result for a session (cached, fast)."""
        try:
            from rppg_processor import RPPGProcessor
        except ImportError:
            return {"error": "rppg_processor.py not found"}

        proc = RPPG_STORE.get(session_id)
        if proc is None:
            return {"status": "no_data", "bpm": 0, "hrv_ms": 0, "waveform": []}
        return proc.compute()

    # ── WebSocket Live Stream ─────────────────────────────────────────────────

    @app.websocket("/metrics/live/{session_id}")
    async def websocket_live(websocket: WebSocket, session_id: str):
        """Real-time biomechanical stream for dashboard."""
        await websocket.accept()
        WS_CONNECTIONS[session_id].append(websocket)
        print(f"[WS] Client connected to session {session_id[:8]}")
        try:
            while True:
                # Keep alive — client messages are ignored
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                except asyncio.TimeoutError:
                    await websocket.send_text(json.dumps({"type": "ping", "ts": time.time()}))
        except WebSocketDisconnect:
            print(f"[WS] Client disconnected from session {session_id[:8]}")
        except Exception as e:
            print(f"[WS] Error: {e}")
        finally:
            conns = WS_CONNECTIONS.get(session_id, [])
            if websocket in conns:
                conns.remove(websocket)

    @app.websocket("/session/{session_id}/live-stream")
    async def websocket_metadata_stream(websocket: WebSocket, session_id: str):
        """Phase 2: High-speed metadata stream from phone React Native C++ Processors (60fps)."""
        await websocket.accept()
        print(f"[WS-STREAM] Native phone streaming 60fps metadata for {session_id[:8]}")
        
        if session_id not in SESSION_DB:
            # Auto-create dummy session if testing directly
            SESSION_DB[session_id] = {"athlete_id": "test", "sport": "vertical_jump", "status": "active"}
            FRAME_BUFFER[session_id] = []
        
        try:
            from pose_analyzer import PoseAnalyzer
            import types
            
            if session_id not in _POSE_ANALYZERS:
                _POSE_ANALYZERS[session_id] = PoseAnalyzer(sport=SESSION_DB[session_id].get("sport", "vertical_jump"))
            analyzer = _POSE_ANALYZERS[session_id]
            
            while True:
                # Expecting tightly packed JSON array of 33 float dicts {x,y,z,visibility}
                data = await websocket.receive_json()
                points = data.get("points", [])
                if not points or len(points) < 33:
                    continue
                
                # Mock MediaPipe Results struct to feed directly into analyzer
                lms = [types.SimpleNamespace(x=p.get('x', 0), y=p.get('y',0), z=p.get('z',0), visibility=p.get('v', 1.0)) for p in points]
                res = types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=lms))
                
                # O(1) Matrix Kinematics Math
                bio = analyzer.analyze(res)
                
                # Stream <1ms results back to phone
                await websocket.send_json({
                    "form_score": bio.form_score,
                    "phase_space_dm": getattr(bio, 'phase_space_dm', 0),
                    "torsion_error": getattr(bio, 'torsion_error', 0),
                    "dimensionless_jerk": getattr(bio, 'dimensionless_jerk', 0),
                    "phase": getattr(bio, 'phase', 'setup'),
                    "primary_feedback": bio.primary_feedback
                })
                
        except WebSocketDisconnect:
            print(f"[WS-STREAM] Native device disconnected {session_id[:8]}")
        except Exception as e:
            print(f"[WS-STREAM] Error: {e}")

    # ── Dataset ───────────────────────────────────────────────────────────────

    @app.get("/dataset/export", tags=["Dataset"])
    async def export_dataset(format: str = Query(default="csv")):
        csv_path = DATASET_PATH / "training_data.csv"
        if not csv_path.exists():
            raise HTTPException(404, "Dataset not found. Run: python generate_dataset.py")
        if format == "json":
            rows = []
            str_fields = {"session_id", "athlete_id", "sport", "phase_label", "quality_label", "feedback_tag"}
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append({k: v if k in str_fields else float(v) for k, v in row.items()})
            return JSONResponse({"data": rows, "count": len(rows)})
        return FileResponse(csv_path, media_type="text/csv", filename="activebharat_dataset.csv")

    @app.get("/dataset/stats", tags=["Dataset"])
    async def dataset_stats():
        stats_path = DATASET_PATH / "sample_stats.json"
        if not stats_path.exists():
            return {"error": "Run: python generate_dataset.py"}
        with open(stats_path, encoding="utf-8") as f:
            return json.load(f)

    # ── Banner ────────────────────────────────────────────────────────────────

    @app.get("/banner", tags=["Health"])
    async def banner():
        """Returns connection banner for Android app to verify connectivity."""
        return {
            "connected": True,
            "server": "ActiveBharat API v2.0",
            "athletes": len(ATHLETE_DB),
            "sessions_today": sum(
                1 for s in SESSION_DB.values()
                if s.get("started_at", "")[:10] == datetime.utcnow().date().isoformat()
            ),
        }


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Install: pip install fastapi uvicorn pydantic")
    else:
        print("============================================================")
        print("  ActiveBharat REST API")
        print("  http://localhost:8082")
        print("  http://localhost:8082/docs  <- Swagger UI")
        print("============================================================")
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=8082,
            reload=True,
            log_level="info",
        )
