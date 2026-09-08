import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.neural_engine import SpeechRecognizer, SpectrogramExtractor
from backend.assistant_engine import VoiceAssistantEngine
from utils.generate_samples import create_synthetic_speech_wav

app = FastAPI(
    title="Neural Speech Recognition & Voice Assistant API",
    description="Neural Speech-to-Text, Audio Spectrogram Feature Extraction, and AI Voice Assistant",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Generate sample audio if not existing
sample_file_1 = os.path.join(SAMPLES_DIR, "speech_sample_1.wav")
if not os.path.exists(sample_file_1):
    create_synthetic_speech_wav(sample_file_1, duration_sec=3.0)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

recognizer = SpeechRecognizer()
assistant = VoiceAssistantEngine()

class CommandRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main Speech Recognition Web UI dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """API endpoint to receive uploaded WAV audio file and return neural speech transcription + spectrogram."""
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file uploaded.")
            
        result = recognizer.process_wav_bytes(content)
        
        # If transcription succeeded, run voice assistant processing on recognized text
        assistant_res = None
        if result.get("success") and result.get("transcription"):
            assistant_res = assistant.process_command(result["transcription"])
            
        return JSONResponse({
            "status": "success",
            "file_name": file.filename,
            "file_size": len(content),
            "result": result,
            "assistant_response": assistant_res
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Transcription failed: {str(e)}"}
        )

@app.post("/api/command")
async def process_voice_command(req: CommandRequest):
    """API endpoint to execute voice assistant command on transcribed speech text."""
    response = assistant.process_command(req.text)
    return JSONResponse({"status": "success", "data": response})

@app.get("/api/samples")
async def get_sample_files():
    """Returns list of pre-generated sample WAV audio files for instant testing."""
    files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".wav")]
    return JSONResponse({
        "samples": [
            {"name": f, "url": f"/static/samples/{f}"} for f in files
        ]
    })

@app.post("/api/train_demo")
async def run_training_demo():
    """Runs PyTorch Speech Model training demo."""
    from train_speech_model import train_neural_speech_model
    train_neural_speech_model(epochs=3)
    return JSONResponse({
        "status": "success",
        "message": "PyTorch Neural Speech Model epoch training executed successfully!"
    })

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
