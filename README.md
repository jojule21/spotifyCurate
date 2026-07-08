# spotifyCurate — Emotion-Driven Spotify Queue

Point your webcam at your face, and it queues music matching your mood. A Flask web app that detects your dominant facial emotion in real time (OpenCV + DeepFace) and queues tracks from your own top Spotify artists via the Spotify Web API (Spotipy).

## How it works

```
Webcam ──▶ OpenCV frames ──▶ DeepFace emotion analysis (every Nth frame)
                                      │
                                      ▼ dominant emotion (happy, sad, angry, ...)
Spotify OAuth ──▶ your top artists ──▶ emotion→genre mapping ──▶ queue next track
```

- `SpotifyCode/main.py` — the full app: Flask server, Spotify OAuth flow (`FlaskSessionCacheHandler`), a background emotion-tracker thread with proper lock-protected shared state, per-emotion rotation indices so you don't get the same song twice, and track queueing against the active Spotify device.
- `MainCode.py` — the original prototype: webcam feed with a live emotion label overlay. Kept as a minimal demo of the CV pipeline (samples every 10th frame to keep latency low).

## Setup

1. Create an app at [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) and add `http://localhost:5000/callback` as a redirect URI.
2. Copy `.env.example` to `.env` and fill in your credentials.
3. Install and run:

```bash
pip install -r requirements.txt
python SpotifyCode/main.py
```

4. Open `http://localhost:5000`, authorize with Spotify, and make sure Spotify is playing on a device (the queue API needs an active device).
5. Press `q` in the camera window to quit.

## Requirements & notes

- Requires **Spotify Premium** (the queue endpoint) and a webcam.
- Emotion detection runs on sampled frames rather than every frame — this keeps inference latency low enough for a live overlay.
- Scopes used: `user-top-read`, `user-modify-playback-state`.

## Status

Working prototype. Spotify has been deprecating parts of its Web API (e.g., recommendations endpoints), so this app deliberately builds queues from *your own top artists* rather than deprecated recommendation calls. If an endpoint 404s for you, check the [Spotify API changelog](https://developer.spotify.com/documentation/web-api).
