import cv2
import numpy as np
from flask import Flask, request, jsonify, session, send_file, send_from_directory
import os
import time
import random
import secrets
import mimetypes
import uuid

app = Flask(__name__, static_folder='static')
app.secret_key = secrets.token_hex(32)  # Bug 3 fix: secure random secret key

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

SPORT_IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'SportsImage'))

# Bug 5 fix: whitelist allowed sports and levels
ALLOWED_SPORTS = {}

def _scan_allowed_sports():
    """Scan SportsImage directory to build whitelist of allowed sport/level combos."""
    global ALLOWED_SPORTS
    if not os.path.isdir(SPORT_IMAGE_DIR):
        return
    for sport in os.listdir(SPORT_IMAGE_DIR):
        sport_path = os.path.join(SPORT_IMAGE_DIR, sport)
        if os.path.isdir(sport_path):
            levels = []
            for level in os.listdir(sport_path):
                level_path = os.path.join(sport_path, level)
                if os.path.isdir(level_path):
                    levels.append(level)
            if levels:
                ALLOWED_SPORTS[sport] = sorted(levels)

_scan_allowed_sports()

# ─── Persona profiles with base trust scores ─────────────────────────────
PERSONAS = [
    {'id': 1, 'name': 'Dr. Priya Sharma',   'role': 'Senior Archery Coach',   'base_score': 380, 'bio': '12 years coaching national-level archers',              'initials': 'PS'},
    {'id': 2, 'name': 'Rahul Verma',         'role': 'Junior Trainee',         'base_score': 275, 'bio': 'First-year sports science student',                     'initials': 'RV'},
    {'id': 3, 'name': 'Coach Meera Patel',   'role': 'Head of Academy',        'base_score': 415, 'bio': 'Olympic selection committee member',                    'initials': 'MP'},
    {'id': 4, 'name': 'Arjun Reddy',         'role': 'Certified Instructor',   'base_score': 390, 'bio': '8 years competitive archery experience',                'initials': 'AR'},
    {'id': 5, 'name': 'Sneha Kulkarni',      'role': 'Sports Analyst',         'base_score': 345, 'bio': 'Data-driven performance assessment specialist',         'initials': 'SK'},
    {'id': 6, 'name': 'Vikram Singh',        'role': 'New Recruit',            'base_score': 300, 'bio': 'Recently joined the archery program',                   'initials': 'VS'},
    {'id': 7, 'name': 'Ananya Desai',        'role': 'Regional Coach',         'base_score': 360, 'bio': '5 years coaching state-level athletes',                 'initials': 'AD'},
]

TEST_SCORE_DELTA = 40

# Dilation detection threshold — percentage of baseline (e.g. 0.05 = 5%)
# Still used by _compute_reaction_times for reaction-time onset detection.
DILATION_THRESHOLD_PCT = 0.05
# Max fraction of correct images that can show dilation before we call it noise
FALSE_POSITIVE_THRESHOLD = 0.30

# ─── Relative comparison verdict ─────────────────────────────────────────
# Effect size threshold (Cohen's d).  0.5 = "medium effect" — the swapped-
# image pupil readings are noticeably higher than the correct-image readings.
EFFECT_SIZE_THRESHOLD = 0.5

# ─── Facial reaction composite weighting ─────────────────────────────
PUPIL_WEIGHT = 0.75
FACIAL_WEIGHT = 0.25
COMPOSITE_THRESHOLD = 0.5  # same as EFFECT_SIZE_THRESHOLD

# ─── Anatomical sanity constants ─────────────────────────────────────────
MIN_PUPIL_PX = 2.0    # minimum plausible iris radius in pixels
MAX_PUPIL_PX = 20.0   # maximum plausible iris radius in pixels

# ─── Ratio-based measurement constants ───────────────────────────────────
# When frontend sends pupil/iris ratio instead of pixel values
MIN_PUPIL_RATIO = 0.15   # minimum plausible pupil/iris ratio
MAX_PUPIL_RATIO = 0.85   # maximum plausible pupil/iris ratio

# ─── Minimum readings per image ─────────────────────────────────────────
MIN_READINGS_PER_IMAGE = 3

# ─── Iris tracking quality threshold ───────────────────────────────────
IRIS_TRACKING_THRESHOLD = 0.70

# ─── Minimum image coverage for a reliable verdict ─────────────────────
MIN_IMAGE_COVERAGE = 0.5

# ─── Liveness / anti-spoofing constants ──────────────────────────────────
MIN_BLINKS_CALIBRATION = 1
FACE_PRESENCE_THRESHOLD = 0.80  # 80% of frames must have a face

# Demo mode: inject simulated dilation at swapped images so the patent demo
# clearly shows the concept end-to-end.  Set to False for real-world use.
# Mutable at runtime via /api/config toggle so the examiner can switch live.
_demo_mode = os.environ.get('DEMO_MODE', '0').lower() in ('1', 'true', 'yes')
_demo_scenario = 'pass'  # 'pass' = expert spots changes, 'fail' = novice misses changes

def _is_demo_mode():
    return _demo_mode

# Keep module-level name for backward compat with tests that import DEMO_MODE
DEMO_MODE = _demo_mode
DEMO_DILATION_BOOST_MIN = 0.14  # 14-22% variable boost at swapped images
DEMO_DILATION_BOOST_MAX = 0.22
DEMO_DRIFT_PER_IMAGE_MIN = 0.005  # 0.5-1.5% gradual upward drift per image
DEMO_DRIFT_PER_IMAGE_MAX = 0.015

# ─── Reaction-time confidence constants ──────────────────────────────────
FAST_REACTION_MS   = 1500   # anything ≤ this gets full boost
SLOW_REACTION_MS   = 3000   # anything ≥ this gets no boost
CONFIDENCE_BOOST   = 1.15   # max multiplier for very fast reactions
CONFIDENCE_NEUTRAL = 1.0    # no boost / no penalty

# ─── Server-side session store ──────────────────────────────────────────────
# Flask's cookie-based sessions are limited to ~4KB.  Pupil readings from the
# browser easily exceed that, so we keep large data server-side in a dict
# keyed by a small UUID that *is* stored in the cookie.
_store = {}

def _get_store():
    """Return the server-side dict for the current session, creating if needed."""
    sid = session.get('_sid')
    if not sid:
        sid = str(uuid.uuid4())
        session['_sid'] = sid
    if sid not in _store:
        _store[sid] = {}
    return _store[sid]


# Helper to get images based on proficiency
def get_images_for_level(sport, level):
    # Bug 5 fix: validate against whitelist
    if sport not in ALLOWED_SPORTS or level not in ALLOWED_SPORTS.get(sport, []):
        return []
    folder = os.path.join(SPORT_IMAGE_DIR, sport, level)
    if not os.path.isdir(folder):
        return []
    images = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    return images


# Helper to pick a wrong image (simulate)
def randomize_image(img_list):
    modified_list = img_list.copy()
    first, second = random.sample(range(len(img_list)), 2)
    modified_list[first], modified_list[second] = modified_list[second], modified_list[first]
    return [modified_list, first, second]


def _validate_liveness(store):
    """Check that the session demonstrates real human presence."""
    blinks = store.get('blink_count', 0)
    cal_presence = store.get('calibration_face_presence', 0)
    slide_presence = store.get('slideshow_face_presence', 0)

    if blinks < MIN_BLINKS_CALIBRATION:
        return False, f'Insufficient blinks ({blinks}/{MIN_BLINKS_CALIBRATION})'
    if cal_presence < FACE_PRESENCE_THRESHOLD:
        return False, f'Low face presence during calibration ({cal_presence:.0%})'
    if slide_presence < FACE_PRESENCE_THRESHOLD:
        return False, f'Low face presence during slideshow ({slide_presence:.0%})'
    return True, 'Liveness confirmed'


def _compute_reaction_times(pupil_data, wrong_indices, calibration, threshold_pct):
    """For each swapped image, find the time from image onset to first dilation
    reading that exceeds the threshold.

    Returns a dict  {image_index: reaction_time_ms}  (only entries where a
    reaction was detected and timestamps were available).
    """
    threshold_px = calibration * threshold_pct
    reaction_times = {}

    for idx in wrong_indices:
        # Collect readings for this image that carry onset info
        readings_for_image = [
            r for r in pupil_data
            if r.get('index') == idx
               and r.get('timestamp_ms') is not None
               and r.get('image_onset_ms') is not None
        ]
        for r in readings_for_image:
            ps = r.get('pupil_size')
            if ps is not None and abs(ps - calibration) > threshold_px:
                rt = r['timestamp_ms'] - r['image_onset_ms']
                if rt >= 0:
                    reaction_times[idx] = rt
                    break  # first qualifying reading wins
    return reaction_times


def _compute_confidence_weight(reaction_times):
    """Derive a confidence multiplier from reaction times at swapped images.

    Linear interpolation:
      0 ms          → CONFIDENCE_BOOST  (1.15)
      FAST_REACTION_MS (1500) → CONFIDENCE_NEUTRAL (1.0)
      > FAST_REACTION_MS      → CONFIDENCE_NEUTRAL (1.0)

    If no reaction times available → CONFIDENCE_NEUTRAL (no penalty).
    """
    if not reaction_times:
        return CONFIDENCE_NEUTRAL

    weights = []
    for rt in reaction_times.values():
        if rt <= 0:
            weights.append(CONFIDENCE_BOOST)
        elif rt >= FAST_REACTION_MS:
            weights.append(CONFIDENCE_NEUTRAL)
        else:
            # linear interpolation: 0 → BOOST, FAST_REACTION_MS → NEUTRAL
            frac = rt / FAST_REACTION_MS
            w = CONFIDENCE_BOOST + frac * (CONFIDENCE_NEUTRAL - CONFIDENCE_BOOST)
            weights.append(w)

    return sum(weights) / len(weights)


def _compute_verdict(per_image_avg, wrong_indices, num_images):
    """Relative comparison verdict using effect size (Cohen's d).

    Instead of comparing each image against an absolute threshold, we ask:
    "are the pupil sizes at swapped images statistically higher than at
    correct images?"  This is noise-robust because both groups share the
    same noise — only the *difference between groups* matters.

    Returns (verdict, swapped_mean, correct_mean, effect_size).
    """
    correct_indices = [i for i in range(num_images) if i not in wrong_indices]

    swapped_vals = [per_image_avg[i] for i in wrong_indices if i in per_image_avg]
    correct_vals = [per_image_avg[i] for i in correct_indices if i in per_image_avg]

    swapped_mean = float(np.mean(swapped_vals)) if swapped_vals else 0.0
    correct_mean = float(np.mean(correct_vals)) if correct_vals else 0.0

    # Pool all per-image averages to compute spread
    all_vals = [per_image_avg[i] for i in range(num_images) if i in per_image_avg]
    pooled_std = float(np.std(all_vals)) if len(all_vals) > 1 else 0.0

    if pooled_std > 0:
        effect_size = (swapped_mean - correct_mean) / pooled_std
    else:
        # No spread at all — can't distinguish groups
        effect_size = 0.0

    if effect_size >= EFFECT_SIZE_THRESHOLD:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return verdict, round(swapped_mean, 4), round(correct_mean, 4), round(effect_size, 4)


def _compute_facial_verdict(per_image_facial, wrong_indices, num_images):
    """Cohen's d comparison on facial reaction scores (same approach as pupil).

    Returns (facial_effect_size, facial_swapped_mean, facial_correct_mean).
    """
    correct_indices = [i for i in range(num_images) if i not in wrong_indices]

    swapped_vals = [per_image_facial[i] for i in wrong_indices if i in per_image_facial]
    correct_vals = [per_image_facial[i] for i in correct_indices if i in per_image_facial]

    swapped_mean = float(np.mean(swapped_vals)) if swapped_vals else 0.0
    correct_mean = float(np.mean(correct_vals)) if correct_vals else 0.0

    all_vals = [per_image_facial[i] for i in range(num_images) if i in per_image_facial]
    pooled_std = float(np.std(all_vals)) if len(all_vals) > 1 else 0.0

    if pooled_std > 0:
        effect_size = (swapped_mean - correct_mean) / pooled_std
    else:
        effect_size = 0.0

    return round(effect_size, 4), round(swapped_mean, 4), round(correct_mean, 4)


def detect_pupil_dilation_from_frame(image_bytes):
    npimg = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    pupil_sizes = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            _, thresh = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                pupil_contour = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(pupil_contour)
                pupil_sizes.append(radius)

    if pupil_sizes:
        return float(np.mean(pupil_sizes))
    return None


# ─── New endpoints for browser-based demo ───────────────────────────────────

@app.route('/')
def index():
    """Serve the frontend HTML."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static assets."""
    return send_from_directory(app.static_folder, filename)


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """GET: return config. POST with {"demo_mode": true/false, "demo_scenario": "pass"|"fail"}: toggle demo mode/scenario."""
    global _demo_mode, DEMO_MODE, _demo_scenario
    if request.method == 'POST':
        data = request.get_json()
        if data and 'demo_mode' in data:
            _demo_mode = bool(data['demo_mode'])
            DEMO_MODE = _demo_mode
        if data and 'demo_scenario' in data:
            val = str(data['demo_scenario']).lower()
            if val in ('pass', 'fail'):
                _demo_scenario = val
    return jsonify({
        'demo_mode': _is_demo_mode(),
        'demo_scenario': _demo_scenario,
        'effect_size_threshold': EFFECT_SIZE_THRESHOLD,
        'pupil_weight': PUPIL_WEIGHT,
        'facial_weight': FACIAL_WEIGHT,
        'composite_threshold': COMPOSITE_THRESHOLD,
    })


@app.route('/api/persona', methods=['GET'])
def api_persona():
    """Return a random persona and store it in the session."""
    persona = random.choice(PERSONAS)
    store = _get_store()
    store['persona'] = persona
    return jsonify(persona)


@app.route('/api/available_sports', methods=['GET'])
def available_sports():
    """List sports and levels for the dropdown."""
    return jsonify(ALLOWED_SPORTS)


@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    """
    Accept pupil size array from browser-side iris detection.
    Expects JSON: { "pupil_sizes": [12.3, 12.5, ...] }
    """
    data = request.get_json()
    if not data or 'pupil_sizes' not in data:
        return jsonify({'error': 'Missing pupil_sizes array'}), 400

    pupil_sizes = [p for p in data['pupil_sizes'] if isinstance(p, (int, float)) and p > 0]
    if not pupil_sizes:
        return jsonify({'error': 'No valid pupil sizes provided'}), 400

    avg_pupil = float(np.mean(pupil_sizes))

    # Detect measurement type from request
    measurement_type = data.get('measurement_type', 'pixel')

    if measurement_type == 'ratio':
        # Ratio-based: pupil/iris ratio (typically 0.3-0.7)
        if avg_pupil < MIN_PUPIL_RATIO:
            return jsonify({'error': f'Pupil ratio too small ({avg_pupil:.3f}). '
                            f'Minimum is {MIN_PUPIL_RATIO}. Move closer to the camera.'}), 400
        if avg_pupil > MAX_PUPIL_RATIO:
            return jsonify({'error': f'Pupil ratio too large ({avg_pupil:.3f}). '
                            f'Maximum is {MAX_PUPIL_RATIO}. Move farther from the camera.'}), 400
    else:
        # Pixel-based: original bounds
        if avg_pupil < MIN_PUPIL_PX:
            return jsonify({'error': f'Pupil size too small ({avg_pupil:.1f}px). '
                            f'Minimum is {MIN_PUPIL_PX}px. Move closer to the camera.'}), 400
        if avg_pupil > MAX_PUPIL_PX:
            return jsonify({'error': f'Pupil size too large ({avg_pupil:.1f}px). '
                            f'Maximum is {MAX_PUPIL_PX}px. Move farther from the camera.'}), 400

    store = _get_store()
    store['calibration_pupil_size'] = avg_pupil
    store['measurement_type'] = measurement_type

    # Liveness fields (optional — old clients may omit them)
    blink_count = data.get('blink_count')
    if blink_count is not None:
        store['blink_count'] = int(blink_count)
    face_presence_ratio = data.get('face_presence_ratio')
    if face_presence_ratio is not None:
        store['calibration_face_presence'] = float(face_presence_ratio)

    return jsonify({'calibration_pupil_size': avg_pupil, 'num_samples': len(pupil_sizes)})


@app.route('/api/submit_pupil_data', methods=['POST'])
def api_submit_pupil_data():
    """
    Batch submit all slideshow pupil readings from the browser.
    Expects JSON: { "readings": [ {"index": 0, "pupil_size": 12.5}, ... ] }
    """
    data = request.get_json()
    if not data or 'readings' not in data:
        return jsonify({'error': 'Missing readings array'}), 400

    readings = data['readings']
    pupil_data = []
    for r in readings:
        idx = r.get('index')
        ps = r.get('pupil_size')
        if idx is not None:
            entry = {
                'index': int(idx),
                'pupil_size': float(ps) if ps is not None else None,
                'timestamp': time.time(),
            }
            # Preserve client-side timing for reaction-time analysis
            if r.get('timestamp_ms') is not None:
                entry['timestamp_ms'] = float(r['timestamp_ms'])
            if r.get('image_onset_ms') is not None:
                entry['image_onset_ms'] = float(r['image_onset_ms'])
            # Preserve detection method for quality filtering
            if r.get('detection_method') is not None:
                entry['detection_method'] = r['detection_method']
            pupil_data.append(entry)

    store = _get_store()
    store['pupil_data'] = pupil_data

    # Liveness: slideshow face presence ratio
    face_presence_ratio = data.get('face_presence_ratio')
    if face_presence_ratio is not None:
        store['slideshow_face_presence'] = float(face_presence_ratio)

    # Iris tracking quality: what fraction of readings came from real MediaPipe
    iris_tracking_ratio = data.get('iris_tracking_ratio')
    if iris_tracking_ratio is not None:
        store['iris_tracking_ratio'] = float(iris_tracking_ratio)

    # Measurement type (ratio vs pixel)
    measurement_type = data.get('measurement_type')
    if measurement_type is not None:
        store['measurement_type'] = measurement_type

    # Detection stats: how many readings came from each method
    detection_stats = data.get('detection_stats')
    if detection_stats is not None:
        store['detection_stats'] = detection_stats

    # Facial reaction readings (optional — old clients may omit)
    facial_readings = data.get('facial_readings')
    if facial_readings is not None:
        store['facial_data'] = facial_readings
    facial_baseline = data.get('facial_baseline')
    if facial_baseline is not None:
        store['facial_baseline'] = float(facial_baseline)

    return jsonify({'received': len(pupil_data)})


@app.route('/api/trust_score', methods=['GET'])
def api_trust_score():
    """
    Compute trust score from dilation data.
    Returns verdict, trust score, per-image data, and swapped image indices.
    """
    store = _get_store()
    calibration = store.get('calibration_pupil_size')
    wrong_indices = store.get('wrong_index', [])
    pupil_data = store.get('pupil_data', [])
    images = store.get('slideshow_images', [])

    if calibration is None:
        return jsonify({'error': 'No calibration data'}), 400
    if not pupil_data:
        return jsonify({'error': 'No pupil data submitted'}), 400

    # ── Liveness gate ──
    liveness_passed, liveness_reason = _validate_liveness(store)
    if not liveness_passed:
        liveness_resp = {
            'verdict': 'LIVENESS FAILED',
            'liveness_passed': False,
            'liveness_reason': liveness_reason,
            'demo_mode': _is_demo_mode(),
            'calibration_pupil_size': calibration,
            'wrong_indices': wrong_indices,
            'image_labels': [f"Image {i+1}" for i in range(len(images))],
            'per_image_sizes': [],
            'dilation_percentages': [],
            'detected_count': 0,
            'total_wrong': len(wrong_indices),
            'false_positives': 0,
            'total_correct': 0,
            'false_positive_ratio': 0.0,
            'swapped_mean': 0.0,
            'correct_mean': 0.0,
            'effect_size': 0.0,
            'reaction_times': {},
            'confidence_weight': 1.0,
            'avg_reaction_time_ms': None,
            'measurement_type': store.get('measurement_type', 'pixel'),
            'detection_quality': 'LOW',
            'detection_stats': store.get('detection_stats', {}),
            'facial_effect_size': 0.0,
            'facial_swapped_mean': 0.0,
            'facial_correct_mean': 0.0,
            'composite_score': 0.0,
            'per_image_facial_scores': [],
            'has_facial_data': False,
        }
        persona = store.get('persona')
        if persona:
            liveness_resp['persona'] = {
                **persona,
                'trust_score': persona['base_score'] - TEST_SCORE_DELTA,
                'score_delta': -TEST_SCORE_DELTA,
            }
        return jsonify(liveness_resp)

    # Filter out random_fallback readings — these are noise, not real measurements
    pupil_data = [r for r in pupil_data if r.get('detection_method') != 'random_fallback']

    # Aggregate: average pupil size per image index
    per_image = {}
    for d in pupil_data:
        idx = d['index']
        ps = d['pupil_size']
        if ps is not None:
            per_image.setdefault(idx, []).append(ps)

    # Remove entries with fewer than MIN_READINGS_PER_IMAGE readings
    per_image = {idx: sizes for idx, sizes in per_image.items()
                 if len(sizes) >= MIN_READINGS_PER_IMAGE}

    if not per_image:
        return jsonify({'error': 'Insufficient pupil readings per image '
                        f'(minimum {MIN_READINGS_PER_IMAGE} per image required)'}), 400

    # ── INCONCLUSIVE check: too few images with enough valid readings ──
    num_images = len(images)
    coverage = len(per_image) / num_images if num_images > 0 else 0
    if not _is_demo_mode() and coverage < MIN_IMAGE_COVERAGE:
        inconclusive_resp = {
            'verdict': 'INCONCLUSIVE',
            'liveness_passed': True,
            'liveness_reason': 'Liveness confirmed',
            'inconclusive_reason': (f'Insufficient data coverage ({coverage:.0%} of images have enough readings). '
                                    'Camera could not reliably track pupils for most images.'),
            'demo_mode': False,
            'calibration_pupil_size': calibration,
            'wrong_indices': wrong_indices,
            'image_labels': [f"Image {i+1}" for i in range(num_images)],
            'per_image_sizes': [],
            'dilation_percentages': [],
            'detected_count': 0,
            'total_wrong': len(wrong_indices),
            'false_positives': 0,
            'total_correct': 0,
            'false_positive_ratio': 0.0,
            'swapped_mean': 0.0,
            'correct_mean': 0.0,
            'effect_size': 0.0,
            'reaction_times': {},
            'confidence_weight': 1.0,
            'avg_reaction_time_ms': None,
            'measurement_type': store.get('measurement_type', 'pixel'),
            'detection_quality': 'LOW',
            'detection_stats': store.get('detection_stats', {}),
            'facial_effect_size': 0.0,
            'facial_swapped_mean': 0.0,
            'facial_correct_mean': 0.0,
            'composite_score': 0.0,
            'per_image_facial_scores': [],
            'has_facial_data': False,
        }
        persona = store.get('persona')
        if persona:
            inconclusive_resp['persona'] = {
                **persona,
                'trust_score': persona['base_score'],
                'score_delta': 0,
            }
        return jsonify(inconclusive_resp)

    per_image_avg = {}
    for idx, sizes in per_image.items():
        per_image_avg[idx] = float(np.mean(sizes))

    # Demo mode: inject simulated dilation so the patent demo clearly shows
    # the concept.  Uses varied jitter, temporal drift, and variable boost
    # to produce natural-looking biological data rather than mechanical charts.
    if _is_demo_mode():
        num_images = len(images)
        # Temporal drift: gradual upward trend simulating pupil adaptation
        drift_per_image = random.uniform(DEMO_DRIFT_PER_IMAGE_MIN, DEMO_DRIFT_PER_IMAGE_MAX)

        # Pre-compute a unique "personality" offset for each correct image
        # so the bar chart looks like real biological variation — some images
        # cause slightly larger pupils (bright/exciting), others slightly smaller.
        image_personality = {}
        for i in range(num_images):
            if i not in wrong_indices:
                # Each correct image gets a fixed offset: some positive, some negative
                # Range: roughly -4% to +5% of baseline — enough to be visually distinct
                image_personality[i] = calibration * random.uniform(-0.04, 0.05)

        for i in range(num_images):
            drift = calibration * drift_per_image * i
            if i not in wrong_indices:
                # Apply personality + small random noise so each bar is unique
                personality = image_personality.get(i, 0)
                noise = calibration * random.gauss(0, 0.012)
                per_image_avg[i] = calibration + personality + drift + noise
            else:
                per_image_avg[i] = calibration  # placeholder, overwritten below

        # Swapped images: boost (PASS scenario) or baseline (FAIL scenario)
        for idx in wrong_indices:
            drift = calibration * drift_per_image * idx
            if _demo_scenario == 'fail':
                # Novice: no boost — swapped images look just like correct images
                personality = calibration * random.uniform(-0.04, 0.05)
                noise = calibration * random.gauss(0, 0.012)
                per_image_avg[idx] = calibration + personality + drift + noise
            else:
                # Expert: variable boost at swapped images (14-22% range)
                boost_pct = random.uniform(DEMO_DILATION_BOOST_MIN, DEMO_DILATION_BOOST_MAX)
                boost = calibration * boost_pct
                extra_jitter = calibration * random.uniform(-0.02, 0.03)
                per_image_avg[idx] = calibration + boost + drift + extra_jitter

        # Inject variation into raw pupil_data readings so reaction-time
        # charts show natural-looking onset patterns
        injected_data = []
        for i in range(num_images):
            target_avg = per_image_avg[i]
            base_onset = i * 4000.0  # approximate onset per image
            for r_idx in range(4):  # 4 readings per image
                ts = base_onset + 800 + r_idx * 250  # after settle window
                # Swapped images: gradual ramp-up from baseline to dilated
                if i in wrong_indices and r_idx == 0:
                    reading_val = calibration + (target_avg - calibration) * 0.3
                elif i in wrong_indices and r_idx == 1:
                    reading_val = calibration + (target_avg - calibration) * 0.7
                else:
                    # Per-reading scatter: ±2% of baseline for realistic noise
                    reading_val = target_avg + calibration * random.gauss(0, 0.02)
                injected_data.append({
                    'index': i,
                    'pupil_size': reading_val,
                    'timestamp': time.time(),
                    'timestamp_ms': ts,
                    'image_onset_ms': base_onset,
                })
            # Update per_image entry from injected readings
            readings_for_i = [r['pupil_size'] for r in injected_data if r['index'] == i]
            per_image_avg[i] = float(np.mean(readings_for_i))

        # Replace pupil_data with injected data for reaction-time calculation
        pupil_data = injected_data

    # Compute dilation percentages
    dilation_pct = {}
    for idx, avg in per_image_avg.items():
        if calibration > 0:
            dilation_pct[idx] = round(((avg - calibration) / calibration) * 100, 2)
        else:
            dilation_pct[idx] = 0.0

    # ── Relative comparison verdict ──
    num_images = len(images)
    verdict, swapped_mean, correct_mean, effect_size = _compute_verdict(
        per_image_avg, wrong_indices, num_images)

    # ── Reaction-time confidence ──
    reaction_times = _compute_reaction_times(
        pupil_data, wrong_indices, calibration, DILATION_THRESHOLD_PCT)
    confidence_weight = _compute_confidence_weight(reaction_times)
    avg_reaction_time_ms = (
        round(sum(reaction_times.values()) / len(reaction_times), 1)
        if reaction_times else None
    )

    # ── Facial reaction composite scoring ──
    facial_data = store.get('facial_data', [])
    has_facial_data = len(facial_data) > 0

    # Demo mode: inject synthetic facial data
    if _is_demo_mode() and not has_facial_data:
        facial_data = []
        for i in range(num_images):
            if i in wrong_indices and _demo_scenario != 'fail':
                score = random.uniform(0.35, 0.55)
            else:
                score = random.uniform(0.02, 0.08)
            facial_data.append({
                'index': i,
                'facial_score': score,
                'timestamp_ms': i * 4000.0 + 1500,
            })
        has_facial_data = True

    # Aggregate facial scores per image
    per_image_facial = {}
    if has_facial_data:
        facial_per_image_raw = {}
        for fd in facial_data:
            idx = fd.get('index')
            fs = fd.get('facial_score')
            if idx is not None and fs is not None:
                facial_per_image_raw.setdefault(idx, []).append(fs)
        for idx, scores in facial_per_image_raw.items():
            per_image_facial[idx] = float(np.mean(scores))

    # Compute facial verdict and composite
    facial_effect_size = 0.0
    facial_swapped_mean = 0.0
    facial_correct_mean = 0.0
    composite_score = 0.0
    ordered_facial = []

    if has_facial_data and per_image_facial:
        facial_effect_size, facial_swapped_mean, facial_correct_mean = \
            _compute_facial_verdict(per_image_facial, wrong_indices, num_images)
        composite_score = round(
            PUPIL_WEIGHT * effect_size + FACIAL_WEIGHT * facial_effect_size, 4)
        # Use composite verdict when facial data present
        if composite_score >= COMPOSITE_THRESHOLD:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        for i in range(num_images):
            ordered_facial.append(round(per_image_facial.get(i, 0.0), 4))
    else:
        # No facial data — pupil-only verdict (already computed above)
        composite_score = round(effect_size, 4)
        for i in range(num_images):
            ordered_facial.append(0.0)

    # Legacy per-image detection counts (kept for charts / backward compat)
    threshold_px = calibration * DILATION_THRESHOLD_PCT
    detected_count = 0
    for idx in wrong_indices:
        avg_at_wrong = per_image_avg.get(idx)
        if avg_at_wrong is not None and abs(avg_at_wrong - calibration) > threshold_px:
            detected_count += 1
    total_wrong = len(wrong_indices)

    correct_indices = [i for i in range(num_images) if i not in wrong_indices]
    false_positives = 0
    for idx in correct_indices:
        avg_at_correct = per_image_avg.get(idx)
        if avg_at_correct is not None and abs(avg_at_correct - calibration) > threshold_px:
            false_positives += 1
    total_correct = len(correct_indices)
    false_positive_ratio = false_positives / total_correct if total_correct > 0 else 0

    # ── Detection quality assessment ──
    detection_stats = store.get('detection_stats', {})
    canvas_count = detection_stats.get('canvas_pupil', 0)
    landmark_count = detection_stats.get('landmark_iris', 0)
    fallback_count = detection_stats.get('fallback', 0)
    total_det = canvas_count + landmark_count + fallback_count
    real_count = canvas_count + landmark_count

    if total_det > 0 and canvas_count / total_det > 0.70:
        detection_quality = 'HIGH'
    elif total_det > 0 and real_count / total_det > 0.50:
        detection_quality = 'MEDIUM'
    else:
        detection_quality = 'LOW'

    # ── Iris tracking quality gate (live mode only) ──
    # In live mode, if most readings came from fallback (not real iris tracking),
    # the data is unreliable — override verdict to prevent false results.
    iris_tracking_ratio = store.get('iris_tracking_ratio')
    iris_warning = None
    if not _is_demo_mode() and iris_tracking_ratio is not None:
        if iris_tracking_ratio < IRIS_TRACKING_THRESHOLD:
            iris_warning = (f'Low iris tracking quality ({iris_tracking_ratio:.0%} real). '
                           'Result may be unreliable — webcam could not track pupils accurately.')
            if verdict in ('PASS', 'FAIL'):
                verdict = 'LOW CONFIDENCE'

    # Build per-image labels
    image_labels = [f"Image {i+1}" for i in range(num_images)]

    ordered_sizes = []
    ordered_dilation = []
    for i in range(num_images):
        ordered_sizes.append(per_image_avg.get(i, calibration))
        ordered_dilation.append(dilation_pct.get(i, 0.0))

    response = {
        'verdict': verdict,
        'liveness_passed': True,
        'demo_mode': _is_demo_mode(),
        'iris_tracking_ratio': iris_tracking_ratio,
        'iris_warning': iris_warning,
        'calibration_pupil_size': calibration,
        'wrong_indices': wrong_indices,
        'image_labels': image_labels,
        'per_image_sizes': ordered_sizes,
        'dilation_percentages': ordered_dilation,
        'detected_count': detected_count,
        'total_wrong': total_wrong,
        'false_positives': false_positives,
        'total_correct': total_correct,
        'false_positive_ratio': round(false_positive_ratio, 4),
        'swapped_mean': swapped_mean,
        'correct_mean': correct_mean,
        'effect_size': effect_size,
        'reaction_times': reaction_times,
        'confidence_weight': round(confidence_weight, 4),
        'avg_reaction_time_ms': avg_reaction_time_ms,
        'measurement_type': store.get('measurement_type', 'pixel'),
        'detection_quality': detection_quality,
        'detection_stats': detection_stats,
        'facial_effect_size': facial_effect_size,
        'facial_swapped_mean': facial_swapped_mean,
        'facial_correct_mean': facial_correct_mean,
        'composite_score': composite_score,
        'per_image_facial_scores': ordered_facial,
        'has_facial_data': has_facial_data,
    }

    persona = store.get('persona')
    if persona:
        delta = TEST_SCORE_DELTA if verdict == 'PASS' else -TEST_SCORE_DELTA
        response['persona'] = {
            **persona,
            'trust_score': persona['base_score'] + delta,
            'score_delta': delta,
        }

    return jsonify(response)


# ─── Original endpoints (bug-fixed) ────────────────────────────────────────

@app.route('/calibrate_pupil', methods=['POST'])
def calibrate_pupil():
    """
    Accepts multiple frames before the slideshow to calibrate the average pupil size.
    Expects form-data: multiple files with key 'frame'.
    """
    files = request.files.getlist('frame')
    pupil_sizes = []
    for file in files:
        image_bytes = file.read()
        pupil_size = detect_pupil_dilation_from_frame(image_bytes)
        if pupil_size:
            pupil_sizes.append(pupil_size)
    if pupil_sizes:
        avg_pupil = float(np.mean(pupil_sizes))
        store = _get_store()
        store['calibration_pupil_size'] = avg_pupil
        return jsonify({'calibration_pupil_size': avg_pupil})
    else:
        return jsonify({'error': 'No valid pupil data found for calibration'}), 400


@app.route('/start_slideshow', methods=['POST'])
def start_slideshow():
    """
    Starts the slideshow based on coach proficiency.
    Expects JSON: { "sport": "Archery", "level": "Basic" }
    """
    data = request.get_json()
    sport = data.get('sport', 'Archery')
    level = data.get('level', 'Basic')

    images = get_images_for_level(sport, level)
    if not images:
        return jsonify({'error': 'No images found for this level'}), 400

    randomize_image_list, first, second = randomize_image(images)

    store = _get_store()
    store['original_images'] = images
    store['slideshow_images'] = randomize_image_list
    store['wrong_index'] = [first, second]
    store['pupil_data'] = []
    store['wrong_image_start_time'] = None
    store['wrong_image_pupil'] = None

    return jsonify({'num_images': len(randomize_image_list)})


@app.route('/get_image/<int:index>', methods=['GET'])
def track_pupil(index):
    """
    Returns the image at the given index in the slideshow.
    """
    store = _get_store()
    images = store.get('slideshow_images', [])
    wrong_indices = store.get('wrong_index', [])
    if not (0 <= index < len(images)):
        return jsonify({'error': 'Invalid index'}), 400

    if index in wrong_indices:
        store.setdefault('wrong_image_start_times', {})[str(index)] = time.time()

    image_path = images[index]
    # Bug 4 fix: use mimetypes.guess_type() instead of hardcoded image/jpeg
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    return send_file(image_path, mimetype=mime_type)


@app.route('/get_learning_image/<int:index>', methods=['GET'])
def get_learning_image(index):
    """
    Returns the image at the given index from the original (un-swapped) image list.
    Used during the learning pass so the user sees images in correct order.
    """
    store = _get_store()
    images = store.get('original_images', [])
    if not (0 <= index < len(images)):
        return jsonify({'error': 'Invalid index'}), 400

    image_path = images[index]
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    return send_file(image_path, mimetype=mime_type)


@app.route('/detect_pupil', methods=['POST'])
def detect_pupil():
    """
    Accepts a single frame (image) as form-data with key 'frame' and the current image index.
    """
    if 'frame' not in request.files or 'index' not in request.form:
        return jsonify({'error': 'No frame or index provided'}), 400
    file = request.files['frame']
    image_bytes = file.read()
    index = int(request.form['index'])
    pupil_size = detect_pupil_dilation_from_frame(image_bytes)

    store = _get_store()
    pupil_data = store.get('pupil_data', [])
    pupil_data.append({'index': index, 'pupil_size': pupil_size, 'timestamp': time.time()})
    store['pupil_data'] = pupil_data

    # Bug 2 fix: use `in` check against list instead of `==` against list
    if index in store.get('wrong_index', []):
        if store.get('wrong_image_pupil') is None:
            store['wrong_image_pupil'] = pupil_size
            store['wrong_image_time'] = time.time()

    return jsonify({'pupil_size': pupil_size})


@app.route('/slideshow_results', methods=['GET'])
def slideshow_results():
    """
    Returns the pupil dilation data for the wrong image(s).
    """
    store = _get_store()
    pupil_data = store.get('pupil_data', [])
    wrong_indices = store.get('wrong_index', [])
    wrong_pupil = store.get('wrong_image_pupil')
    wrong_time = store.get('wrong_image_time')
    wrong_start_times = store.get('wrong_image_start_times', {})
    calibration_pupil_size = store.get('calibration_pupil_size', None)

    prev_pupils = []
    for idx in wrong_indices:
        prev_pupil = None
        for d in pupil_data:
            if d['index'] == idx - 1:
                prev_pupil = d['pupil_size']
                break
        prev_pupils.append(prev_pupil)

    response_times = []
    for idx in wrong_indices:
        response_time = None
        start_time = wrong_start_times.get(str(idx))
        for d in pupil_data:
            if d['index'] == idx + 1 and start_time:
                response_time = d['timestamp'] - start_time
                break
        response_times.append(response_time)

    after_pupil_sizes = [d['pupil_size'] for d in pupil_data if d['pupil_size'] is not None]
    avg_after_pupil = float(np.mean(after_pupil_sizes)) if after_pupil_sizes else None

    return jsonify({
        'wrong_image_index': wrong_indices,
        'pupil_size_on_wrong_image': wrong_pupil,
        'change_in_pupil_size': [
            (wrong_pupil - prev_pupil) if (wrong_pupil and prev_pupil) else None
            for prev_pupil in prev_pupils
        ],
        'response_times': response_times,
        'calibration_pupil_size': calibration_pupil_size,
        'average_pupil_size_after_slideshow': avg_after_pupil
    })


@app.route('/pupil_recognition_result', methods=['GET'])
def pupil_recognition_result():
    """
    Returns 'yes' if the pupil size changed at the time of looking at swapped (wrong) images.
    """
    store = _get_store()
    calibration_pupil_size = store.get('calibration_pupil_size', None)
    wrong_indices = store.get('wrong_index', [])
    pupil_data = store.get('pupil_data', [])

    if calibration_pupil_size is None or not wrong_indices or not pupil_data:
        return jsonify({'result': 'no', 'reason': 'Missing calibration or pupil data'})

    wrong_pupil_sizes = []
    for idx in wrong_indices:
        for d in pupil_data:
            if d['index'] == idx and d['pupil_size'] is not None:
                wrong_pupil_sizes.append(d['pupil_size'])
                break

    if not wrong_pupil_sizes:
        return jsonify({'result': 'no', 'reason': 'No pupil data for wrong images'})

    # Bug 1 fix: use threshold-based comparison instead of exact float equality
    threshold_px = calibration_pupil_size * DILATION_THRESHOLD_PCT
    detected = any(abs(p - calibration_pupil_size) > threshold_px for p in wrong_pupil_sizes)
    result = "yes" if detected else "no"

    return jsonify({
        'result': result,
        'calibration_pupil_size': calibration_pupil_size,
        'wrong_pupil_sizes': wrong_pupil_sizes,
    })


if __name__ == '__main__':
    # Bug 6 fix: use env var for debug mode
    debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    app.run(debug=debug, host='0.0.0.0', port=5000)
