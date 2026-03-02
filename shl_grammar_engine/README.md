# SHL Grammar Engine (Real-Time English Communication Assessment)

Production-grade spoken English communication assessment app inspired by SHL / Versant / Pearson style screening.

## Features
- Browser microphone capture using **WebRTC** (`streamlit-webrtc`).
- Real-time ASR with **faster-whisper** (`large-v3`).
- Grammar scoring with **`textattack/bert-base-uncased-CoLA`** transformer.
- Live results: transcript, grammar score (0–5), confidence.
- FastAPI backend endpoints (model singletons are lazy-loaded via FastAPI dependencies):
  - `POST /transcribe`
  - `POST /score`
  - `POST /analyze`
- Evaluation pipeline:
  - WER (Speech Accent Archive audio)
  - Grammar accuracy (CoLA validation set)

## Project Structure
```text
shl_grammar_engine/
  backend/
    main.py
    asr.py
    grammar.py
    analyzer.py
    evaluate.py
  frontend/
    app.py
  models/
  requirements.txt
  Dockerfile
  README.md
```

## Local Setup
```bash
cd shl_grammar_engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Backend (FastAPI)
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### Run Frontend (Streamlit)
```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/app.py --server.port 8501
```

Open `http://localhost:8501`.

### Run Tests
```bash
pytest -q
```

## API Examples
### Transcribe audio
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@sample.wav"
```

### Score text
```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{"text": "She goes to office every day."}'
```

### Analyze audio end-to-end
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "audio=@sample.wav"
```

## Dataset Evaluation

### 1) Speech Accent Archive (Kaggle)
Dataset: https://www.kaggle.com/datasets/rtatman/speech-accent-archive

Expected extracted layout:
```text
<accent_root>/
  speakers_all.csv
  recordings/
    <filename>.wav
```

### 2) Run evaluation
```bash
python -m backend.evaluate --accent-root /path/to/speech-accent-archive --max-audio 20 --cola-samples 500
```

Output includes:
- `wer_accent_archive`
- `grammar_accuracy_cola`

## Deployment

## Docker (Backend)
```bash
cd shl_grammar_engine
docker build -t shl-grammar-engine .
docker run --rm -p 8000:8000 shl-grammar-engine
```

Run frontend separately or via another container with:
```bash
BACKEND_URL=http://<backend-host>:8000 streamlit run frontend/app.py
```

## Streamlit Cloud
1. Push repo to GitHub.
2. In Streamlit Cloud, create app for `frontend/app.py`.
3. Set environment variable `BACKEND_URL` to your deployed FastAPI URL.
4. Add secrets/env if using auth/rate limiting.

## HuggingFace Spaces
- Use **Streamlit Space**.
- App entrypoint: `frontend/app.py`.
- Include `requirements.txt`.
- Set `BACKEND_URL` as Space variable if backend is external.
- For single-space setup, optionally deploy backend separately on HF Inference Endpoint / Railway / Render.

## Production Notes
- Use GPU for `large-v3` for low latency.
- Enable request throttling and authentication before public exposure.
- Add persistent logging (e.g., OpenTelemetry + Prometheus/Grafana).
- Add CI: lint, tests, model smoke checks.
- Consider async queueing for high concurrency.
