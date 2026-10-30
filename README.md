# Dragonwing Quest Assistant

A voice assistant that can see. You wear a Meta Quest 3S, press a button and ask a question out loud, and the headset streams your voice and what you are looking at to a Qualcomm Dragonwing IQ-9075 board. The board does all of the AI on-device and speaks the answer back. No cloud, fully offline.

The Quest is only the eyes, ears and mouth. Every model runs on the IQ-9075, most of it on the Hexagon NPU.

## What happens when you ask a question

1. You press the left menu button on the Quest and speak.
2. The headset records your voice and grabs the current passthrough camera frame.
3. Both are sent to the board over your local Wi-Fi.
4. The board transcribes your speech, feeds the question and the image to a vision language model, and turns the answer into speech.
5. The spoken answer plays back in the headset, and the text shows on a floating card.

Typical round trip is a couple of seconds.

## The pieces

**On the Quest (Unity app)**

Built on [QuestCameraKit](https://github.com/xrdevrob/QuestCameraKit) by Robert Rodriguez (xrdevrob), using the ImageLLM sample as the starting point. The passthrough frames come from Meta's Passthrough Camera API.

Main scripts (in `Assets/Samples/5 ImageLLM/Scripts`):

- `SttManager.cs` records the microphone and posts the audio to the box.
- `VoiceCommandHandler.cs` captures the passthrough frame.
- `ImageOpenAIConnector.cs` sends the frame and the question to the box.
- `TtsManager.cs` plays the spoken answer back.
- `SceneCard.cs` draws the floating debug card.

**On the IQ-9075 (box server)**

`server.py` is a FastAPI server on port 8765 with three endpoints:

- `/stt` speech to text, using faster-whisper (base.en, int8).
- `/vlm` the vision language model, Qwen3-VL-4B-Instruct running on the NPU through GenieX.
- `/tts` text to speech, using Piper (en_US-lessac-medium) encoded to MP3.

## Box setup

On the IQ-9075, with the Qualcomm AI stack installed:

```
pip install fastapi uvicorn faster-whisper piper-tts
python -m piper.download_voices en_US-lessac-medium
```

Pull the Qwen3-VL-4B NPU bundle with the Qualcomm AI Hub tooling, then run:

```
python server.py
```

The server listens on `0.0.0.0:8765`.

## Building the Quest app

1. Open the project in Unity 6000.3.12f1 with the Android module.
2. Open `Assets/Samples/5 ImageLLM/ImageLLM.unity`.
3. In Player Settings: Microphone Usage Description set to something like "Used for voice questions", and Allow downloads over HTTP set to Always allowed (the app talks to the box over plain HTTP).
4. In Build Profiles, tick only the ImageLLM scene.
5. Plug in the Quest and Build and Run.

## Known limitations and next steps

- The box address is hard-coded in the Unity scripts. Anyone rebuilding this has to change it to their own board's IP. Automatic discovery on the local network is the obvious next improvement.
- The model files are not in the repo. You get them with the Qualcomm AI Hub export and install commands above.

## Credits

- [QuestCameraKit](https://github.com/xrdevrob/QuestCameraKit) by Robert Rodriguez (xrdevrob) for the passthrough camera access and the ImageLLM sample.
- Qwen3-VL, Whisper, Piper, and the Qualcomm AI Hub tooling for the on-device models.
