import os
import time
import threading
from collections import deque, Counter
from threading import Event

from flask import Flask, redirect, url_for, session, request
from dotenv import load_dotenv
from deepface import DeepFace
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
import cv2

load_dotenv()

# ---------------- Config / Globals ----------------
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
redirect_uri = os.getenv("REDIRECT_URI")
# FIX #5: Load secret key from env so sessions survive restarts
secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(64).hex())
scope = "user-top-read user-modify-playback-state"

_emotion_thread = None
_emotion_thread_stop_event = Event()
_emotion_thread_lock = threading.Lock()

# FIX #2: Lock to protect shared globals accessed by both Flask and tracker threads
_globals_lock = threading.Lock()
user_artists = []  # cached top artists [{id,name,genres,top_track_uri}]
last_idx_per_emotion = {}  # rotation indices per emotion key
current_genre = None  # genre of the last queued song

# ---------------- Flask / Spotify OAuth ----------------
app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key
cache_handler = FlaskSessionCacheHandler(session)

sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=scope,
    cache_handler=cache_handler,
    show_dialog=True
)

# ---------------- Routes ----------------
@app.route('/')
def home():
    if not sp_oauth.validate_token(cache_handler.get_cached_token()):
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    return redirect(url_for('get_artists'))

@app.route('/callback')
def callback():
    token_info = sp_oauth.get_access_token(request.args.get('code'))
    cache_handler.save_token_to_cache(token_info)
    return redirect(url_for('get_artists'))

@app.route('/get_artists')
def get_artists():
    token_info = cache_handler.get_cached_token()
    if not token_info:
        return redirect(url_for('home'))

    sp_user = Spotify(auth=token_info['access_token'])

    # FIX #2: Acquire lock before writing to shared globals
    with _globals_lock:
        global user_artists, last_idx_per_emotion
        user_artists = []
        try:
            top_artists = sp_user.current_user_top_artists(limit=10, time_range='long_term').get('items', [])
            for a in top_artists:
                aid = a.get('id')
                name = a.get('name')
                genres = a.get('genres', []) or []
                top_uri = None
                try:
                    top_tracks = sp_user.artist_top_tracks(aid, country='US').get('tracks', [])
                    if top_tracks:
                        top_uri = top_tracks[0].get('uri')
                except Exception:
                    top_uri = None

                user_artists.append({
                    'id': aid,
                    'name': name,
                    'genres': genres,
                    'top_track_uri': top_uri
                })

        except Exception as e:
            print("Error fetching top artists:", e)

        # Initialize rotation indices
        for emo in ["happy", "sad", "angry", "neutral", "fear", "disgust", "surprise"]:
            last_idx_per_emotion.setdefault(emo, 0)

    # Start emotion tracking thread
    # FIX #1: Pass sp_oauth so the thread can refresh the token when it expires
    global _emotion_thread, _emotion_thread_stop_event
    with _emotion_thread_lock:
        if _emotion_thread is None or not _emotion_thread.is_alive():
            _emotion_thread_stop_event.clear()
            _emotion_thread = threading.Thread(
                target=start_emotion_tracking,
                args=(token_info, sp_oauth, _emotion_thread_stop_event),
                daemon=True
            )
            _emotion_thread.start()
        else:
            print("Emotion tracker already running")

    return "Cached top artists and started emotion tracker."

@app.route('/stop_tracker')
def stop_tracker():
    global _emotion_thread_stop_event, _emotion_thread
    _emotion_thread_stop_event.set()
    if _emotion_thread:
        _emotion_thread.join(timeout=5)
    _emotion_thread = None
    return "Stopped tracker"

# ---------------- Token refresh helper ----------------
def get_fresh_spotify_client(token_info, oauth):
    """Return a Spotify client with a refreshed token if the current one is expired."""
    # FIX #1: Validate and refresh before each use in the background thread
    if oauth.is_token_expired(token_info):
        print("Token expired — refreshing...")
        token_info = oauth.refresh_access_token(token_info['refresh_token'])
    return Spotify(auth=token_info['access_token']), token_info

# ---------------- Emotion-based Track Queue ----------------
def play_track_for_emotion(sp_user, emotion):
    global user_artists, last_idx_per_emotion, current_genre

    # FIX #2: Acquire lock before reading/writing shared globals
    with _globals_lock:
        if not user_artists:
            print("No cached artists available.")
            return

        candidates = []

        # FIX #3: Handle current_genre being None on first run — skip genre filtering
        #         and fall through to the full artist list immediately.
        if current_genre is not None:
            for a in user_artists:
                genres = a.get('genres', [])
                if not genres or not a.get('top_track_uri'):
                    continue

                emotion_key = emotion.lower()
                # Neutral/Happy/Sad/Angry -> prefer same genre for continuity
                if emotion_key in ["neutral", "happy", "sad", "angry"] and current_genre in genres:
                    candidates.append(a)
                # Disgust/Fear/Surprise -> prefer genre change
                elif emotion_key in ["disgust", "fear", "surprise"] and current_genre not in genres:
                    candidates.append(a)

        # Fallback: use all artists with a valid track URI
        if not candidates:
            candidates = [a for a in user_artists if a.get('top_track_uri')]

        if not candidates:
            print("No candidates to queue.")
            return

        key = emotion.lower()
        idx = last_idx_per_emotion.get(key, 0) % len(candidates)
        choice = candidates[idx]
        last_idx_per_emotion[key] = (idx + 1) % len(candidates)

    track_uri = choice.get('top_track_uri')
    if track_uri:
        try:
            sp_user.add_to_queue(track_uri)
            # FIX #2: Update current_genre under lock
            with _globals_lock:
                current_genre = choice['genres'][0] if choice.get('genres') else None
            print(f"[{time.strftime('%H:%M:%S')}] Queued '{choice.get('name')}' "
                  f"(emotion: {emotion}, genre: {current_genre})")
        except Exception as e:
            # FIX #7: Log the full Spotify error, not just a generic message
            print(f"Failed to queue track '{choice.get('name')}': {e}")

# ---------------- Headless Emotion Tracker ----------------
def start_emotion_tracking(token_info, oauth, stop_event):
    # FIX #1: Don't build the client once — refresh it when the token expires
    sp_user, token_info = get_fresh_spotify_client(token_info, oauth)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)

    sample_interval = 1.0
    window_seconds = 60
    log_interval = 5.0

    samples = deque()
    last_sample_time = 0
    last_queue_time = time.time()
    last_log_time = time.time()
    # FIX #1: Track when to next check token validity (check every 5 minutes)
    last_token_refresh = time.time()
    token_check_interval = 300

    try:
        while not stop_event.is_set():
            ret, img = cap.read()
            if not ret:
                print("Webcam error — could not read frame.")
                break

            now = time.time()

            # FIX #1: Periodically refresh the Spotify token before it expires
            if now - last_token_refresh >= token_check_interval:
                sp_user, token_info = get_fresh_spotify_client(token_info, oauth)
                last_token_refresh = now

            if now - last_sample_time >= sample_interval:
                # FIX #4: Use `except Exception` instead of bare `except`
                try:
                    res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                    if isinstance(res, list) and len(res) > 0:
                        emotion = res[0].get('dominant_emotion', 'neutral')
                    elif isinstance(res, dict):
                        emotion = res.get('dominant_emotion', 'neutral')
                    else:
                        emotion = 'neutral'
                except Exception as e:
                    print(f"DeepFace analysis failed: {e}")
                    emotion = 'neutral'

                samples.append((emotion, now))
                last_sample_time = now

                # Remove samples outside the rolling window
                while samples and now - samples[0][1] > window_seconds:
                    samples.popleft()

            # Log dominant emotion every 5 seconds
            if now - last_log_time >= log_interval and samples:
                emotions_only = [s[0] for s in samples]
                dominant = Counter(emotions_only).most_common(1)[0][0]
                print(f"[{time.strftime('%H:%M:%S')}] Current dominant emotion: {dominant}")
                last_log_time = now

            # Queue a track every 60 seconds based on dominant emotion
            if now - last_queue_time >= window_seconds and samples:
                emotions_only = [s[0] for s in samples]
                dominant = Counter(emotions_only).most_common(1)[0][0]
                play_track_for_emotion(sp_user, dominant)
                last_queue_time = now
                samples.clear()

            time.sleep(0.05)

    finally:
        # FIX #8: cap.release() is always called via finally, even on crash
        cap.release()
        print("Webcam released, tracker stopped.")

# ---------------- Run Flask ----------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5500, debug=True, use_reloader=False)
