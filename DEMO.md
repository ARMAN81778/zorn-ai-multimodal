# ZORN SIGHT — 60–90 Second Demo Script

## Setup (before judging starts)

```bash
pip install -r requirements.txt
python demo.py
```

Confirm the model has already been downloaded once so load time doesn't eat
into the demo window. Have `examples/test.jpg` and a short `examples/test.mp4`
(5–10 seconds) ready to upload.

## Script

**0:00 – 0:10 — Launch**
Open the Gradio page. Point at the Hardware / Runtime / Status panel:
"This is running fully local — SmolVLM2-500M-Instruct, detected on this
Intel CPU, backend auto-selected."

**0:10 – 0:20 — Text**
Type in Chat: *"What can you see?"* — show the normal conversational
response and the performance line (backend, latency).

**0:20 – 0:35 — Image**
Upload `examples/test.jpg`. Ask: *"Describe this image and identify the most
important objects."* Show the real model response.

**0:35 – 0:55 — Video**
Upload a short video. Ask: *"What is happening in this video?"* Point out:
- the response
- the Evidence panel: sampled frame indices and timestamps
- the number of frames actually processed (not every frame — smart
  sampling)

**0:55 – 1:05 — Evidence follow-up**
Ask: *"Why did you reach that conclusion?"* Point back at the same evidence
frames used to ground the answer.

**1:05 – 1:15 — Voice**
Click "🔊 Speak Response." Play the generated audio file.

**1:15 – 1:30 — Benchmark**
Open the Benchmark tab, click "Run Benchmark." Read the actual measured
backend, latency, and memory values out loud — emphasize these are measured
on this machine right now, not invented.

**Closing line**
"One compact multimodal model. One optimized runtime. Multiple hardware
targets — Intel CPU today through OpenVINO, with a documented Qualcomm QNN
deployment path for Snapdragon NPUs."
