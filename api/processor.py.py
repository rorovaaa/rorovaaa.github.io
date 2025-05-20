import os
import json
import cv2
import numpy as np
from http import HTTPStatus
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as rt
from tensorflow.keras.models import load_model
from collections import deque
import logging
import uuid

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CONFIG = {
    "RSL": {
        "model_path": os.path.join(os.path.dirname(__file__), "..", "models", "rsl_model.onnx"),
        "labels": os.path.join(os.path.dirname(__file__), "..", "labels", "rsl_labels.txt"),
        "input_shape": (3, 32, 224, 224),
        "provider": "CPUExecutionProvider",
        "window_size": 32
    },
    "ASL": {
        "model_path": os.path.join(os.path.dirname(__file__), "..", "models", "asl_model.onnx"),
        "labels": os.path.join(os.path.dirname(__file__), "..", "labels", "asl_labels.txt"),
        "input_shape": (28, 28, 1),
        "window_size": 1
    }
}

sessions: Dict[str, Dict[str, Any]] = {}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_model(model_type: str):
    config = MODEL_CONFIG[model_type]
    
    if model_type == "RSL":
        session = rt.InferenceSession(
            config["model_path"],
            providers=[config["provider"]]
        )
        labels = load_labels(config["labels"], delimiter='\t')
        return {"model": session, "labels": labels, "type": "onnx"}
    
    model = load_model(config["model_path"])
    labels = load_labels(config["labels"], delimiter=' ')
    return {"model": model, "labels": labels, "type": "keras"}

def load_labels(path: str, delimiter: str = '\t') -> Dict[int, str]:
    labels = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) >= 2:
                    labels[int(parts[0])] = parts[1]
        return labels
    except Exception as e:
        logging.error(f"Error loading labels: {str(e)}")
        return {}

def process_frame(session_id: str, frame_data: bytes):
    try:
        session_data = sessions[session_id]
        
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if session_data['model_type'] == "RSL":
            processed = preprocess_rsl(frame)
        else:
            processed = preprocess_asl(frame)
        
        session_data['buffer'].append(processed)
        
        if len(session_data['buffer']) >= session_data['window_size']:
            return make_prediction(session_data)
            
        return None
        
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        raise

def preprocess_rsl(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame.astype(np.float32) / 255.0

def preprocess_asl(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (28, 28))
    return np.expand_dims(frame, axis=-1).astype(np.float32) / 255.0

def make_prediction(session_data):
    model = session_data['model']
    buffer = np.array(session_data['buffer'])
    
    if session_data['model_type'] == "RSL":
        buffer = np.transpose(buffer, (3, 0, 1, 2))
        inputs = {model.get_inputs()[0].name: np.expand_dims(buffer, 0)}
        outputs = model.run(None, inputs)[0][0]
    else:
        outputs = model.predict(buffer, verbose=0)[0]
    
    return format_output(outputs, session_data['labels'], session_data['threshold'])

def format_output(predictions, labels, threshold):
    probs = softmax(predictions)
    top_indices = np.argsort(probs)[-3:][::-1]
    
    results = []
    for idx in top_indices:
        if probs[idx] > threshold:
            results.append({
                "label": labels.get(idx, "Unknown"),
                "confidence": float(probs[idx])
            })
    
    return {"gestures": results[:3]}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.post("/api/init")
async def initialize_session(request: Request):
    try:
        params = await request.json()
        session_id = str(uuid.uuid4())
        
        sessions[session_id] = {
            "model": initialize_model(params['model_type']),
            "buffer": deque(maxlen=MODEL_CONFIG[params['model_type']]["window_size"]),
            "model_type": params['model_type'],
            "threshold": params.get('threshold', 0.7),
            "window_size": MODEL_CONFIG[params['model_type']]["window_size"],
            "labels": load_labels(MODEL_CONFIG[params['model_type']]['labels'])
        }
        
        return JSONResponse(
            content={"session_id": session_id},
            status_code=HTTPStatus.CREATED
        )
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )

@app.post("/api/process")
async def process_image(session_id: str, file: UploadFile = File(...)):
    try:
        frame_data = await file.read()
        result = process_frame(session_id, frame_data)
        
        if result:
            return JSONResponse(content=result)
            
        return JSONResponse(
            content={"status": "buffer_not_full"},
            status_code=HTTPStatus.ACCEPTED
        )
        
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )

@app.delete("/api/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleaned"}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse(os.path.join("static", "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
