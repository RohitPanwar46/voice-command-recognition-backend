# Voice Command Recognition API

FastAPI backend serving the trained CNN model. Accepts a short audio
recording and returns the predicted command.

## 1. Copy your model files in

Before deploying (or running locally), copy these 3 files from your
training project into `app/model/`:

```
app/model/cnn_voice_command_model.keras
app/model/classes.json
app/model/feature_stats.json
```

## 2. Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Test it:
```bash
curl -X POST -F "file=@some_recording.wav" http://localhost:8000/predict
```

## 3. Deploy to Render

1. Push this folder to a GitHub repo.
2. On Render: New -> Web Service -> connect the repo.
3. Render will detect `render.yaml` and the Dockerfile automatically.
   (If not, manually set: Environment = Docker, Dockerfile path = ./Dockerfile)
4. Once deployed, note your API URL, e.g. `https://voice-command-api.onrender.com`
   - You'll need this for the frontend's `NEXT_PUBLIC_API_URL`.

## Endpoints

- `GET /health` -> `{"status": "ok"}`
- `POST /predict` -> multipart form with a `file` field (any audio format
  ffmpeg can decode: webm, wav, ogg, mp3...) -> returns:
  ```json
  {
    "label": "four",
    "confidence": 0.85,
    "top3": [
      {"label": "four", "confidence": 0.85},
      {"label": "unknown", "confidence": 0.10},
      {"label": "five", "confidence": 0.02}
    ]
  }
  ```

## Note on CORS

`app/main.py` currently allows all origins (`"*"`) for easy testing.
Once your Vercel frontend is deployed, tighten this to your actual domain.
