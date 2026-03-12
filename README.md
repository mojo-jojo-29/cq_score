# CQ Score — Credibility Quotient for Sports Professionals

A scoring system that quantifies the credibility of sports professionals (athletes, coaches, officials) by combining **document-based credibility (CQ Score)** with **biometric expertise verification (Trust Score)**.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [CQ Score Module](#cq-score-module)
- [Trust Score Module — Pupil Dilation Assessment](#trust-score-module--pupil-dilation-assessment)
  - [How It Works](#how-it-works)
  - [Two-Pass Flow](#two-pass-flow)
  - [Liveness / Anti-Spoofing](#liveness--anti-spoofing)
  - [Demo Mode](#demo-mode)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running Locally](#running-locally)
- [Running Tests](#running-tests)
- [API Reference](#api-reference)
- [Configuration](#configuration)

---

## Overview

The system assigns a **CQ Score (0–620)** based on verifiable profile data, documents, certifications, and engagement. A subset of this is the **Trust Score (0–100)**, which uses involuntary pupil dilation responses to verify whether a person genuinely recognises sport-specific image sequences — something that cannot be faked.

**Key idea:** An expert in a sport will have an involuntary pupil dilation response when they see images displayed in an incorrect order, because their brain recognises the anomaly even before they consciously process it. A non-expert has no such reference and shows no measurable dilation change.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   CQ Score (0–620)                │
│                                                   │
│  Profile Data ─────────────────── up to 105 pts   │
│  Documents (personal, edu) ────── up to 100 pts   │
│  Certifications (gold/silver/..)─ up to 250 pts   │
│  Experience (coach/official/vol)─ up to 150 pts   │
│  Daily Engagement Streak ──────── up to 35 pts    │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │        Trust Score (0–100)                  │  │
│  │  Pupil dilation biometric verification      │  │
│  │  Learning Pass → Test Pass → Analysis       │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

## CQ Score Module

**File:** `script/cq_score.py` (Python), `script/cq_score.js` (JavaScript)

Calculates a composite credibility score from weighted categories:

| Category | Max Points | Components |
|---|---|---|
| Profile fields | 105 | name, DOB, email, phone, gender, org, sport, Aadhar, PIN |
| My People / My Place | 30 | social graph, geographic verification |
| Stats | 30 | basic, intermediate, advanced proficiency levels |
| Personal documents | 50 | up to 5 documents |
| Education documents | 50 | up to 5 documents |
| Certifications | 200 | gold, silver, bronze, participation (up to 5 each) |
| Experience | 150 | coaching, officiating, volunteering (up to 5 each) |
| Daily app visits | 35 | streak-based, 5 pts/day, max 7 days |

**Score cap:** 620

## Trust Score Module — Pupil Dilation Assessment

**Directory:** `script/trust_score/`

A Flask web application that uses webcam-based iris tracking (MediaPipe FaceLandmarker) to measure involuntary pupil dilation as a biometric signal of domain expertise.

### How It Works

1. **Image Sequencing:** Sport-specific images are loaded in a known correct order (e.g., archery technique progression from stance to release).
2. **Swap Introduction:** Two images in the sequence are randomly swapped.
3. **Pupil Measurement:** As the user views each image, their pupil size is tracked via the webcam.
4. **Relative Comparison (Effect Size):** Instead of comparing each image against an absolute threshold, the system uses Cohen's d effect size to measure whether swapped images produce statistically higher pupil sizes than correct images. This is noise-robust because both groups share the same measurement noise — only the *difference between groups* matters.
5. **Reaction Time Analysis:** The speed at which the pupil responds at swapped images provides an additional confidence signal. Faster involuntary reactions indicate stronger domain recognition.
6. **Scoring:** If the effect size exceeds the threshold (d >= 0.5, a "medium effect"), the user is verified as an expert.

### Two-Pass Flow

```
Select Sport/Level → Learning Pass → Transition → Test Pass → Processing → Results
```

| Phase | Duration | What Happens |
|---|---|---|
| **Select** | User-driven | Pick sport and proficiency level |
| **Learning Pass** | ~4s per image | Images shown in correct order; user memorises the sequence; webcam collects baseline pupil data + blink/face tracking |
| **Transition** | 3 seconds | "Now we'll test your recall. Some images may be in a different order." |
| **Test Pass** | ~4s per image | Same images with 2 swapped; pupil tracking at 250ms intervals |
| **Processing** | ~1.5 seconds | Submits pupil data, computes trust score |
| **Results** | — | Verdict badge, trust score circle, per-image bar chart, dilation % line chart |

The learning pass replaces the previous blank-camera calibration. By showing images in the correct order first, every user builds a mental reference, making dilation spikes at swapped images during the test pass stronger and more measurable.

### Liveness / Anti-Spoofing

The system guards against spoofing with three checks:

| Check | Threshold | Measured During |
|---|---|---|
| Blink count | >= 2 blinks | Learning pass |
| Face presence (calibration) | >= 80% of frames | Learning pass |
| Face presence (slideshow) | >= 80% of frames | Test pass |

If any check fails, the verdict is **LIVENESS FAILED** with a trust score of 0.

Blink detection uses the Eye Aspect Ratio (EAR) method with MediaPipe face landmarks, sampled every ~150ms during the learning pass render loop for reliable detection of quick blinks.

### Demo Mode

By default (`DEMO_MODE=0`), real pupil data from the webcam is used. Set `DEMO_MODE=1` to inject a variable 14-22% dilation boost at swapped image positions with natural-looking temporal drift and jitter, making the concept clearly visible without high-precision eye-tracking hardware.

## Project Structure

```
cq_score-main/
├── README.md
├── requirements.txt              # Python dependencies
├── run.py                        # Entry point for trust score web app
├── .gitignore
│
├── script/
│   ├── cq_score.py               # CQ score calculation (Python)
│   ├── cq_score.js               # CQ score calculation (JavaScript)
│   │
│   └── trust_score/
│       ├── pupil_dilation.py     # Flask backend — all API endpoints
│       ├── SportsImage/          # Sport image datasets
│       │   └── {Sport}/{Level}/  # e.g., Archery/Basic/, Archery/Intermediate/
│       └── static/
│           ├── index.html        # Single-page frontend
│           ├── app.js            # Frontend logic (phases, iris tracking, charts)
│           └── style.css         # Styling
│
└── test/
    ├── test_cqScore.py           # CQ score unit tests
    └── test_pupil_dilation.py    # Trust score API + integration tests
```

## Setup

```bash
# Clone and enter the project
cd cq_score-main

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- Flask >= 3.0.0
- NumPy >= 1.24.0
- OpenCV (headless) >= 4.8.0

**Browser requirements:** A modern browser with webcam access (Chrome/Edge recommended for MediaPipe GPU support).

## Running Locally

```bash
# Default (port 5000)
python3 run.py

# Custom port (e.g., 5001 if 5000 is taken by AirPlay on macOS)
PORT=5001 python3 run.py

# With debug mode
FLASK_DEBUG=1 python3 run.py

# Disable demo dilation boost
DEMO_MODE=0 python3 run.py
```

Open `http://localhost:<port>` in your browser.

## Running Tests

```bash
python3 -m pytest test/ -v
```

All tests run without a browser or webcam — they exercise the Flask API endpoints directly.

## API Reference

### Selection & Slideshow

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Serve the frontend |
| `GET /api/available_sports` | GET | List sports and proficiency levels |
| `POST /start_slideshow` | POST | Start session: loads images, creates swap, stores original + swapped lists |
| `GET /get_learning_image/<index>` | GET | Serve image from **original** (correct) order — used in learning pass |
| `GET /get_image/<index>` | GET | Serve image from **swapped** order — used in test pass |

### Calibration & Pupil Data

| Endpoint | Method | Description |
|---|---|---|
| `POST /api/calibrate` | POST | Submit baseline pupil sizes + liveness data (blink count, face presence) |
| `POST /api/submit_pupil_data` | POST | Batch submit all slideshow pupil readings + face presence |
| `POST /detect_pupil` | POST | Submit single frame for server-side pupil detection (legacy) |

### Results

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/trust_score` | GET | Compute and return trust score, verdict, per-image data, dilation percentages |
| `GET /slideshow_results` | GET | Raw dilation data for swapped images (legacy) |
| `GET /pupil_recognition_result` | GET | Simple yes/no dilation detection (legacy) |

### Request/Response Examples

**POST /start_slideshow**
```json
// Request
{ "sport": "Archery", "level": "Basic" }

// Response
{ "num_images": 12 }
```

**POST /api/calibrate**
```json
// Request
{ "pupil_sizes": [10.2, 10.5, ...], "blink_count": 4, "face_presence_ratio": 0.92 }

// Response
{ "calibration_pupil_size": 10.35, "num_samples": 30 }
```

**GET /api/trust_score**
```json
{
  "verdict": "PASS",
  "liveness_passed": true,
  "demo_mode": false,
  "calibration_pupil_size": 10.35,
  "wrong_indices": [3, 7],
  "image_labels": ["Image 1", "Image 2", ...],
  "per_image_sizes": [10.3, 10.4, ...],
  "dilation_percentages": [0.5, 1.2, ...],
  "detected_count": 2,
  "total_wrong": 2,
  "swapped_mean": 11.82,
  "correct_mean": 10.35,
  "effect_size": 1.2345,
  "reaction_times": {"3": 450, "7": 620},
  "confidence_weight": 1.12,
  "avg_reaction_time_ms": 535.0
}
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Server port |
| `FLASK_DEBUG` | `0` | Enable Flask debug mode |
| `DEMO_MODE` | `0` | Inject simulated dilation at swapped images |

| Constant (in code) | Value | Description |
|---|---|---|
| `DILATION_THRESHOLD_PCT` | `0.05` (5%) | Minimum pupil change to count as dilation |
| `EFFECT_SIZE_THRESHOLD` | `0.5` | Cohen's d threshold for PASS verdict |
| `MIN_BLINKS_CALIBRATION` | `2` | Minimum blinks required for liveness |
| `FACE_PRESENCE_THRESHOLD` | `0.80` | Minimum face presence ratio (80%) |
| `MIN_PUPIL_PX` | `2.0` | Minimum plausible iris radius in pixels |
| `MAX_PUPIL_PX` | `20.0` | Maximum plausible iris radius in pixels |
| `MIN_READINGS_PER_IMAGE` | `3` | Minimum pupil readings required per image |
| `DEMO_DILATION_BOOST_MIN/MAX` | `0.14`/`0.22` | Variable dilation boost range in demo mode |
| `IMAGE_DURATION_MS` | `4000` | Time per image in both passes (4s) |
| `EAR_THRESHOLD` | `0.25` | Eye Aspect Ratio threshold for blink detection |
