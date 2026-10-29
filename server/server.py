# server.py  -  offline voice-assistant server for the IQ-9075
# Run with:  uvicorn server:app --host 0.0.0.0 --port 8765
#
# Pipeline: Quest sends voice + camera frame ->
#   /stt  faster-whisper (CPU)      -> transcription
#   /vlm  Qwen3-VL-8B via GenieX (NPU) -> answer about the frame
#   /tts  Piper (CPU)               -> spoken answer
#
# Install first:
#   pip install fastapi uvicorn python-multipart faster-whisper opencv-python numpy
#   sudo apt install ffmpeg
#   python -m piper.download_voices en_US-lessac-medium   # run in this folder
#   (geniex must already be installed in this venv)

import os, base64, subprocess, time, wave
import numpy as np, cv2
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import PlainTextResponse, Response
from faster_whisper import WhisperModel
from geniex import AutoModelForVision2Seq
import re 

app = FastAPI()

PIPER_MODEL = "en_US-lessac-medium.onnx"
VLM_REPO = "ai-hub-models/Qwen3-VL-4B-Instruct"
#VLM_REPO = "ai-hub-models/Gemma-4-E4B-it"

# ---- Speech-to-text: fast CPU (CTranslate2, int8). "tiny.en" if you want it faster.
print("Loading Whisper...")
stt_model = WhisperModel("base.en", device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 8)

# ---- Vision-language model on the NPU via GenieX. Loaded once; this is the slow step.
print(f"Loading VLM {VLM_REPO} on NPU (this takes a while)...")
_t = time.time()
vlm = AutoModelForVision2Seq.from_pretrained(VLM_REPO, device_map="npu")
print(f"VLM loaded in {time.time() - _t:.1f} s.")


# def run_whisper(path):
#     subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
#                     "-ar", "16000", "-ac", "1", "ready.wav"], check=True)
#     with wave.open("ready.wav", "rb") as w:
#         audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
#     t = time.time()
#     #segments, _ = stt_model.transcribe(audio, beam_size=1)
#     segments, _ = stt_model.transcribe(audio, beam_size=1, vad_filter=True)
#     text = " ".join(s.text for s in segments).strip()
#     print(f"[stt] {time.time() - t:.2f}s  {text}")
#     return text

def run_whisper(path):
    # Whisper wants 16 kHz mono. The headset records at 48 kHz, so resample first.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                    "-ar", "16000", "-ac", "1", "ready.wav"], check=True)
    with wave.open("ready.wav", "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0

    # Boost quiet recordings so faint speech is not thrown away as silence.
    peak = float(np.abs(audio).max())
    audio = audio / max(peak, 1e-6) * 0.9

    t = time.time()
    # Looser VAD threshold so quiet speech still counts as speech.
    segments, _ = stt_model.transcribe(audio, beam_size=1, vad_filter=True,
                                       vad_parameters=dict(threshold=0.3))
    text = " ".join(s.text for s in segments).strip()
    print(f"[stt] {time.time() - t:.2f}s  {text}")
    return text


def run_vlm(img, prompt):
    if img is None:
        return "No image received."
    frame_path = "frame.jpg"
    cv2.imwrite(frame_path, img)

    question = prompt.strip() if prompt and prompt.strip() else "What do you see?"
    instruction = ("You are a helpful assistant describing what the headset camera sees. "
                   "Answer in one or two short, clear sentences. ")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": frame_path},
            {"type": "text", "text": instruction + question},
        ],
    }]
    try:
        text_prompt = vlm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text_prompt = vlm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    output = vlm.generate(text_prompt, images=[frame_path], max_new_tokens=128)
    text = output.text.strip()
    text = re.sub(r"[*#`]", "", text)   # drop markdown so TTS reads cleanly
    return text


@app.post("/stt", response_class=PlainTextResponse)
async def stt(file: UploadFile = File(...)):
    with open("incoming.wav", "wb") as f:
        f.write(await file.read())
    return run_whisper("incoming.wav")


@app.post("/vlm")
async def vlm_endpoint(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "What do you see?")
    img = None
    if data.get("image"):
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data["image"]), np.uint8),
                           cv2.IMREAD_COLOR)
    t = time.time()
    answer = run_vlm(img, prompt)
    print(f"[vlm] {time.time() - t:.2f}s  {answer}")
    return {"text": answer}


@app.post("/tts")
async def tts(req: Request):
    text = (await req.json()).get("text", "")
    subprocess.run(["piper", "--model", PIPER_MODEL, "--output_file", "tts.wav"],
                   input=text.encode(), check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "tts.wav", "tts.mp3"],
                   check=True)
    with open("tts.mp3", "rb") as f:
        return Response(content=f.read(), media_type="audio/mpeg")