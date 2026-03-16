/* ─── Pupil Dilation Trust Assessment — Frontend ─────────────────────────── */
(function () {
    'use strict';

    // ─── DOM references ─────────────────────────────────────────────────────
    const phases = {
        select:     document.getElementById('phase-select'),
        learning:   document.getElementById('phase-learning'),
        transition: document.getElementById('phase-transition'),
        slideshow:  document.getElementById('phase-slideshow'),
        processing: document.getElementById('phase-processing'),
        results:    document.getElementById('phase-results'),
    };

    const sportSelect   = document.getElementById('sport-select');
    const levelSelect   = document.getElementById('level-select');
    const btnBegin      = document.getElementById('btn-begin');

    // Learning phase DOM refs
    const learnImg      = document.getElementById('learning-img');
    const learnVideo    = document.getElementById('learning-video');
    const learnCanvas   = document.getElementById('learning-canvas');
    const learnCtx      = learnCanvas.getContext('2d');
    const learnProgress = document.getElementById('learning-progress');
    const learnCounter  = document.getElementById('learning-counter');

    const slideImg      = document.getElementById('slideshow-img');
    const slideVideo    = document.getElementById('slideshow-video');
    const slideCanvas   = document.getElementById('slideshow-canvas');
    const slideCtx      = slideCanvas.getContext('2d');
    const imageCounter  = document.getElementById('image-counter');
    const slideProgress = document.getElementById('slideshow-progress');
    const livePupil     = document.getElementById('live-pupil');

    // ─── State ──────────────────────────────────────────────────────────────
    let stream = null;
    let faceMesh = null;
    let sportsData = {};
    let numImages = 0;
    let wrongIndices = [];             // swapped image indices from start_slideshow
    let calibrationPupilSize = null;   // stored client-side for stateless POST
    let currentImageIndex = 0;
    let calibrationSamples = [];
    let slideshowReadings = [];
    let trackingInterval = null;
    let learnRafId = null;
    let lastLandmarks = null;         // latest iris landmarks for rendering
    let imageStartTimes = {};         // {imageIndex: performance.now()} for reaction-time tracking
    let realIrisReadings = 0;         // count of readings from actual MediaPipe iris tracking
    let fallbackReadings = 0;         // count of readings from simulated fallback data
    let canvasPupilReadings = 0;      // count of readings from canvas-based pupil detection
    let landmarkFallbackReadings = 0; // count of readings from landmark iris (old method)
    let randomFallbackReadings = 0;   // count of readings from random fallback
    let canvasPupilActive = false;    // whether canvas pupil detection is working
    let cameraResolution = { width: 0, height: 0 }; // actual camera resolution
    let lastDetectionMethod = 'unknown'; // detection method used for the latest reading
    let cameraQuality = 'unknown';       // measured camera quality: 'high', 'medium', 'low'

    // ─── Theme toggle ──────────────────────────────────────────────────────
    function getTheme() {
        return document.documentElement.getAttribute('data-theme') || 'dark';
    }
    function setTheme(t) {
        document.documentElement.setAttribute('data-theme', t);
        localStorage.setItem('theme', t);
        // Update Chart.js defaults for new charts
        if (t === 'light') {
            Chart.defaults.color = '#1a1a2e';
            Chart.defaults.borderColor = '#e0e0e0';
        } else {
            Chart.defaults.color = '#8b949e';
            Chart.defaults.borderColor = '#21262d';
        }
    }
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            setTheme(getTheme() === 'dark' ? 'light' : 'dark');
        });
    }
    // Chart color helpers — read from CSS vars so charts match the active theme
    function chartColor(varName) {
        return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    }

    // ─── Persona state ─────────────────────────────────────────────────────
    let currentPersona = null;

    // ─── Facial reaction state ────────────────────────────────────────────
    let facialBaseline = { eyeOpen: 0, browHeight: 0, mouthOpen: 0, nosePos: { x: 0, y: 0 } };
    let facialBaselineSamples = [];
    let slideshowFacialReadings = [];

    const CALIBRATION_TARGET = 30;
    const IMAGE_DURATION_MS  = 4000;
    const TRACKING_INTERVAL_FAST = 150; // when canvas pupil detection active
    const TRACKING_INTERVAL_SLOW = 250; // when using full MediaPipe only
    let TRACKING_INTERVAL  = 250;       // current interval (adjusted at runtime)
    const CAL_SAMPLE_INTERVAL = 500;
    const SETTLE_MS          = 800;  // skip readings within this window after image change (light adaptation)

    // ─── EMA smoothing & outlier rejection for real detection ────────────
    const EMA_ALPHA = 0.3;
    const OUTLIER_STD_MULT = 3.0;
    let emaPupil = null;               // exponential moving average
    let pupilHistory = [];             // recent readings for std calculation
    const PUPIL_HISTORY_SIZE = 10;

    // ─── Liveness / anti-spoofing state ─────────────────────────────────
    const EAR_FALLBACK_THRESHOLD = 0.25; // absolute fallback before calibration
    const EAR_BLINK_RATIO = 0.80;       // blink = EAR drops below 80% of baseline
    const EAR_BASELINE_SAMPLES = 20;     // frames to collect for baseline
    const LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144];
    const RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380];

    let blinkCount = 0;
    let eyesClosed = false;
    let earBaseline = null;              // adaptive: mean open-eye EAR
    let earBaselineBuf = [];             // buffer for computing baseline
    let earThreshold = EAR_FALLBACK_THRESHOLD; // updated once baseline is set

    // Face presence counters (calibration)
    let calFaceFrames  = 0;
    let calTotalFrames = 0;

    // Face presence counters (slideshow)
    let slideFaceFrames  = 0;
    let slideTotalFrames = 0;

    // ─── Phase management ───────────────────────────────────────────────────
    function showPhase(name) {
        Object.values(phases).forEach(el => el.classList.remove('active'));
        phases[name].classList.add('active');
    }

    // ─── Webcam ─────────────────────────────────────────────────────────────
    async function startCamera(videoEl) {
        if (stream) {
            videoEl.srcObject = stream;
            return;
        }

        // Progressive camera fallback chain
        const constraints = [
            { video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } },
            { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } },
            { video: { facingMode: 'user' } },
            { video: true },
        ];

        let lastErr = null;
        for (const constraint of constraints) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraint);
                lastErr = null;
                break;
            } catch (err) {
                lastErr = err;
                // NotAllowedError won't be fixed by trying lower resolution
                if (err.name === 'NotAllowedError' || err.name === 'NotFoundError') break;
            }
        }

        if (lastErr || !stream) {
            const err = lastErr || new Error('Camera unavailable');
            let msg = 'Camera access failed.';
            if (err.name === 'NotAllowedError') {
                msg = 'Camera permission denied. Please allow camera access in your browser settings and reload.';
            } else if (err.name === 'NotFoundError') {
                msg = 'No camera found. Please connect a webcam and reload the page.';
            } else if (err.name === 'NotReadableError') {
                msg = 'Camera is in use by another application. Close it and reload.';
            } else {
                msg = `Camera error: ${err.message || err.name}`;
            }
            showCameraError(msg);
            throw err;
        }
        videoEl.srcObject = stream;

        // Read actual camera resolution and update tracking
        const track = stream.getVideoTracks()[0];
        if (track && track.getSettings) {
            const settings = track.getSettings();
            cameraResolution.width = settings.width || 0;
            cameraResolution.height = settings.height || 0;
            console.log(`Camera resolution: ${cameraResolution.width}x${cameraResolution.height}`);
        }

        // Update actualVideoWidth once video metadata loads
        videoEl.addEventListener('loadedmetadata', () => {
            actualVideoWidth = videoEl.videoWidth || 1280;
            cameraResolution.width = cameraResolution.width || videoEl.videoWidth;
            cameraResolution.height = cameraResolution.height || videoEl.videoHeight;
        }, { once: true });

        // Lock exposure/gain after a short warmup so auto-exposure settles first
        // Skip on mobile — causes video to darken on many phone cameras
        if (!(/Android|iPhone|iPad|iPod/i.test(navigator.userAgent))) {
            setTimeout(() => lockExposure(), 2000);
        }
    }

    function showCameraError(msg) {
        const overlay = document.getElementById('camera-error-overlay');
        const msgEl = document.getElementById('camera-error-msg');
        if (overlay && msgEl) {
            msgEl.textContent = msg;
            overlay.style.display = 'flex';
        } else {
            showToast(msg, 'error');
        }
    }

    async function retryCameraAccess() {
        const overlay = document.getElementById('camera-error-overlay');
        if (overlay) overlay.style.display = 'none';
        stream = null;
        try {
            await startCamera(learnVideo);
            showToast('Camera connected successfully!', 'success');
        } catch (_) { /* showCameraError already called */ }
    }

    /** Lock webcam exposure & white balance to prevent auto-adjustment
     *  during the assessment.  Falls back silently if unsupported. */
    async function lockExposure() {
        if (!stream) return;
        try {
            const track = stream.getVideoTracks()[0];
            const caps = track.getCapabilities ? track.getCapabilities() : {};
            const constraints = { advanced: [{}] };
            let hasLock = false;

            if (caps.exposureMode && caps.exposureMode.includes('manual')) {
                constraints.advanced[0].exposureMode = 'manual';
                hasLock = true;
            }
            if (caps.whiteBalanceMode && caps.whiteBalanceMode.includes('manual')) {
                constraints.advanced[0].whiteBalanceMode = 'manual';
                hasLock = true;
            }

            if (hasLock) {
                await track.applyConstraints(constraints);
                console.log('Webcam exposure/WB locked to manual');
            } else {
                console.log('Manual exposure not supported by this camera');
            }
        } catch (e) {
            console.warn('Could not lock exposure:', e);
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
    }

    // ─── Drawing helpers ────────────────────────────────────────────────────

    // MediaPipe face oval landmark indices (forms a contour around the face)
    const FACE_OVAL_IDX = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
    ];
    const BG_BLUR_PX = 12;

    /** Draw a green circle + center dot for one eye */
    function drawEyeMarker(ctx, cx, cy, radius) {
        // Soft glow
        ctx.beginPath();
        ctx.arc(cx, cy, radius + 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(76, 175, 80, 0.2)';
        ctx.fill();

        // Green ring
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = '#4caf50';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Bright center dot
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#4caf50';
        ctx.fill();
    }

    /** Draw iris markers from real MediaPipe landmarks */
    function drawIrisFromLandmarks(ctx, w, h, landmarks) {
        if (!landmarks || landmarks.length < 478) return;

        // Left iris: center 468, ring 469-472
        const lc = landmarks[468];
        const lr = avgLandmarkDist(landmarks, 468, [469, 470, 471, 472]) * w;

        // Right iris: center 473, ring 474-477
        const rc = landmarks[473];
        const rr = avgLandmarkDist(landmarks, 473, [474, 475, 476, 477]) * w;

        drawEyeMarker(ctx, lc.x * w, lc.y * h, Math.max(lr, 5));
        drawEyeMarker(ctx, rc.x * w, rc.y * h, Math.max(rr, 5));
    }

    function avgLandmarkDist(landmarks, centerIdx, ringIndices) {
        const c = landmarks[centerIdx];
        let total = 0;
        for (const ri of ringIndices) {
            const p = landmarks[ri];
            const dx = p.x - c.x;
            const dy = p.y - c.y;
            total += Math.sqrt(dx * dx + dy * dy);
        }
        return total / ringIndices.length;
    }

    // ─── EAR (Eye Aspect Ratio) blink detection ─────────────────────────
    function landmarkDist(a, b) {
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    function computeEAR(landmarks, eyeIndices) {
        const p1 = landmarks[eyeIndices[0]];
        const p2 = landmarks[eyeIndices[1]];
        const p3 = landmarks[eyeIndices[2]];
        const p4 = landmarks[eyeIndices[3]];
        const p5 = landmarks[eyeIndices[4]];
        const p6 = landmarks[eyeIndices[5]];
        const vertical1 = landmarkDist(p2, p6);
        const vertical2 = landmarkDist(p3, p5);
        const horizontal = landmarkDist(p1, p4);
        return (vertical1 + vertical2) / (2.0 * horizontal);
    }

    let earLogCounter = 0;
    let lastBlinkLandmarks = null;  // track to avoid re-processing same landmarks
    function updateBlinkDetection(landmarks) {
        if (!landmarks || landmarks.length < 468) return;
        // Skip if we already processed this exact landmark set
        if (landmarks === lastBlinkLandmarks) return;
        lastBlinkLandmarks = landmarks;
        const leftEAR  = computeEAR(landmarks, LEFT_EYE_IDX);
        const rightEAR = computeEAR(landmarks, RIGHT_EYE_IDX);
        const avgEAR   = (leftEAR + rightEAR) / 2.0;

        // Adaptive baseline: collect EAR samples to set threshold
        if (earBaseline === null) {
            earBaselineBuf.push(avgEAR);
            if (earBaselineBuf.length >= EAR_BASELINE_SAMPLES) {
                // Use median (robust to blinks during calibration)
                const sorted = [...earBaselineBuf].sort((a, b) => a - b);
                earBaseline = sorted[Math.floor(sorted.length * 0.75)]; // 75th percentile = open eyes
                earThreshold = earBaseline * EAR_BLINK_RATIO;
                console.log(`EAR baseline set: ${earBaseline.toFixed(3)}, blink threshold: ${earThreshold.toFixed(3)}`);
            }
        }

        // Log EAR periodically to help debug blink detection on mobile
        if (earLogCounter++ % 30 === 0) {
            console.log(`EAR: ${avgEAR.toFixed(3)} threshold: ${earThreshold.toFixed(3)} baseline: ${earBaseline?.toFixed(3) || 'calibrating'} closed: ${eyesClosed} blinks: ${blinkCount}`);
        }

        if (avgEAR < earThreshold) {
            eyesClosed = true;
        } else if (eyesClosed) {
            // closed → open transition = one blink
            eyesClosed = false;
            blinkCount++;
            console.log(`Blink #${blinkCount} detected (EAR: ${avgEAR.toFixed(3)}, threshold: ${earThreshold.toFixed(3)})`);
            const el = document.getElementById('blink-count');
            if (el) el.textContent = blinkCount;
        }
    }

    // ─── Facial reaction computation ─────────────────────────────────────
    function computeFacialMetrics(landmarks) {
        if (!landmarks || landmarks.length < 478) return null;

        // Eye widening: vertical opening vs horizontal width
        const leftUpperLid = landmarks[159];
        const leftLowerLid = landmarks[145];
        const leftInner = landmarks[33];
        const leftOuter = landmarks[133];
        const rightUpperLid = landmarks[386];
        const rightLowerLid = landmarks[374];
        const rightInner = landmarks[362];
        const rightOuter = landmarks[263];

        const leftVertical = landmarkDist(leftUpperLid, leftLowerLid);
        const leftHorizontal = landmarkDist(leftInner, leftOuter);
        const rightVertical = landmarkDist(rightUpperLid, rightLowerLid);
        const rightHorizontal = landmarkDist(rightInner, rightOuter);
        const eyeOpen = ((leftVertical / (leftHorizontal || 1)) + (rightVertical / (rightHorizontal || 1))) / 2;

        // Eyebrow raise: brow-to-eye distance normalized by face height
        const leftBrow = landmarks[105];
        const rightBrow = landmarks[334];
        const faceTop = landmarks[10];
        const faceChin = landmarks[152];
        const faceHeight = landmarkDist(faceTop, faceChin) || 1;
        const leftBrowDist = landmarkDist(leftBrow, leftInner);
        const rightBrowDist = landmarkDist(rightBrow, rightInner);
        const browHeight = ((leftBrowDist + rightBrowDist) / 2) / faceHeight;

        // Mouth opening: lip separation normalized by face width
        const upperLip = landmarks[13];
        const lowerLip = landmarks[14];
        const faceLeft = landmarks[234];
        const faceRight = landmarks[454];
        const faceWidth = landmarkDist(faceLeft, faceRight) || 1;
        const mouthOpen = landmarkDist(upperLip, lowerLip) / faceWidth;

        // Head movement: nose tip position
        const noseTip = landmarks[1];
        const nosePos = { x: noseTip.x, y: noseTip.y };

        return { eyeOpen, browHeight, mouthOpen, nosePos };
    }

    function computeFacialReaction(landmarks, baseline) {
        const metrics = computeFacialMetrics(landmarks);
        if (!metrics || !baseline || baseline.eyeOpen === 0) return 0;

        // Absolute deviation — captures both directions (e.g. eye widening
        // AND narrowing, brow raising AND furrowing, mouth opening AND pressing)
        function clampScore(current, base) {
            if (base === 0) return 0;
            return Math.max(0, Math.min(Math.abs(current - base) / base, 1));
        }

        const eyeScore = clampScore(metrics.eyeOpen, baseline.eyeOpen);
        const browScore = clampScore(metrics.browHeight, baseline.browHeight);
        const mouthScore = clampScore(metrics.mouthOpen, baseline.mouthOpen);

        // Head movement: displacement from baseline nose position
        const dx = metrics.nosePos.x - baseline.nosePos.x;
        const dy = metrics.nosePos.y - baseline.nosePos.y;
        const headDisplacement = Math.sqrt(dx * dx + dy * dy);
        // Normalize: 0.06 displacement = score 1.0 (less sensitive to natural sway)
        const headScore = Math.max(0, Math.min(headDisplacement / 0.06, 1));

        // Weight facial expressions higher than head movement (noise-prone)
        return eyeScore * 0.3 + browScore * 0.3 + mouthScore * 0.3 + headScore * 0.1;
    }

    function collectFacialBaselineSample(landmarks) {
        const metrics = computeFacialMetrics(landmarks);
        if (metrics) facialBaselineSamples.push(metrics);
    }

    function finalizeFacialBaseline() {
        if (facialBaselineSamples.length === 0) return;
        let sumEye = 0, sumBrow = 0, sumMouth = 0, sumX = 0, sumY = 0;
        for (const s of facialBaselineSamples) {
            sumEye += s.eyeOpen;
            sumBrow += s.browHeight;
            sumMouth += s.mouthOpen;
            sumX += s.nosePos.x;
            sumY += s.nosePos.y;
        }
        const n = facialBaselineSamples.length;
        facialBaseline = {
            eyeOpen: sumEye / n,
            browHeight: sumBrow / n,
            mouthOpen: sumMouth / n,
            nosePos: { x: sumX / n, y: sumY / n },
        };
    }

    /** Draw approximate eye markers when MediaPipe is not available.
     *  Uses fixed ratios of the canvas that roughly correspond to where
     *  a centered face's eyes would be. */
    function drawSimulatedIris(ctx, w, h) {
        const wobble = () => (Math.random() - 0.5) * 2;
        // Typical eye positions for a centered face in a 4:3 frame
        drawEyeMarker(ctx, w * 0.39 + wobble(), h * 0.41 + wobble(), 8);
        drawEyeMarker(ctx, w * 0.61 + wobble(), h * 0.41 + wobble(), 8);
    }

    /**
     * Render a video frame onto a canvas with blurred background,
     * sharp face region, and iris marker overlays.
     */
    function renderFrame(videoEl, ctx, canvas) {
        const w = canvas.width;
        const h = canvas.height;

        if (videoEl.readyState >= 2) {
            if (lastLandmarks && lastLandmarks.length >= 468) {
                // 1. Draw blurred background (skip on mobile — causes frame drops)
                ctx.save();
                if (!isMobile) {
                    ctx.filter = `blur(${BG_BLUR_PX}px)`;
                }
                ctx.drawImage(videoEl, 0, 0, w, h);
                ctx.restore();

                // 2. Clip to face oval and draw sharp face on top
                ctx.save();
                ctx.beginPath();
                for (let i = 0; i < FACE_OVAL_IDX.length; i++) {
                    const lm = lastLandmarks[FACE_OVAL_IDX[i]];
                    const px = lm.x * w;
                    const py = lm.y * h;
                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.closePath();
                ctx.clip();
                ctx.drawImage(videoEl, 0, 0, w, h);
                ctx.restore();
            } else {
                // No landmarks — draw plain frame (no blur without face data)
                ctx.drawImage(videoEl, 0, 0, w, h);
            }
        }

        // Draw iris dots
        if (lastLandmarks) {
            drawIrisFromLandmarks(ctx, w, h, lastLandmarks);
        } else {
            drawSimulatedIris(ctx, w, h);
        }
    }

    // ─── MediaPipe Face Mesh ────────────────────────────────────────────────
    /** Actual video width for iris pixel conversion (updated when camera starts) */
    let actualVideoWidth = 1920;

    function irisRadiusPx(landmarks, centerIdx, ringIndices) {
        const c = landmarks[centerIdx];
        let totalDist = 0;
        for (const ri of ringIndices) {
            const p = landmarks[ri];
            const dx = (p.x - c.x);
            const dy = (p.y - c.y);
            totalDist += Math.sqrt(dx * dx + dy * dy);
        }
        return (totalDist / ringIndices.length) * actualVideoWidth;
    }

    function extractPupilRadius(landmarks) {
        if (!landmarks || landmarks.length < 478) return null;
        const leftR  = irisRadiusPx(landmarks, 468, [469, 470, 471, 472]);
        const rightR = irisRadiusPx(landmarks, 473, [474, 475, 476, 477]);
        return (leftR + rightR) / 2;
    }

    // ─── Canvas-based pupil detection (CORE FIX) ─────────────────────────
    // MediaPipe landmarks 468-477 measure the IRIS boundary, which does NOT
    // change with pupil dilation.  This function crops the eye region from
    // the canvas and uses image processing to find the actual dark pupil.

    /**
     * Detect pupil in one eye by cropping eye ROI from canvas.
     * @param {CanvasRenderingContext2D} sourceCtx - context with video frame drawn
     * @param {number} w - canvas width
     * @param {number} h - canvas height
     * @param {object} corner1 - landmark for inner eye corner
     * @param {object} corner2 - landmark for outer eye corner
     * @param {number} irisR - iris radius in pixels (from MediaPipe) for ratio
     * @returns {{pupilRadiusPx: number, pupilIrisRatio: number}|null}
     */
    function detectPupilInOneEye(sourceCtx, w, h, corner1, corner2, irisR) {
        // Define eye bounding box from corner landmarks with padding
        const x1 = Math.floor(corner1.x * w);
        const y1 = Math.floor(corner1.y * h);
        const x2 = Math.floor(corner2.x * w);
        const y2 = Math.floor(corner2.y * h);
        const eyeW = Math.abs(x2 - x1);
        const eyeH = Math.max(Math.floor(eyeW * 0.6), 10); // eye height ~60% of width
        const padX = Math.floor(eyeW * 0.15);
        const padY = Math.floor(eyeH * 0.3);

        const roiX = Math.max(0, Math.min(x1, x2) - padX);
        const roiY = Math.max(0, Math.floor((y1 + y2) / 2) - Math.floor(eyeH / 2) - padY);
        const roiW = Math.min(eyeW + 2 * padX, w - roiX);
        const roiH = Math.min(eyeH + 2 * padY, h - roiY);

        if (roiW < 8 || roiH < 6) return null;

        // Extract eye ROI pixels
        let imageData;
        try {
            imageData = sourceCtx.getImageData(roiX, roiY, roiW, roiH);
        } catch (e) {
            return null; // security restriction or invalid dimensions
        }
        const pixels = imageData.data;

        // Convert to grayscale
        const gray = new Uint8Array(roiW * roiH);
        for (let i = 0; i < gray.length; i++) {
            const r = pixels[i * 4];
            const g = pixels[i * 4 + 1];
            const b = pixels[i * 4 + 2];
            gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        }

        // 3x3 box blur
        const blurred = new Uint8Array(roiW * roiH);
        for (let y = 1; y < roiH - 1; y++) {
            for (let x = 1; x < roiW - 1; x++) {
                let sum = 0;
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        sum += gray[(y + dy) * roiW + (x + dx)];
                    }
                }
                blurred[y * roiW + x] = Math.round(sum / 9);
            }
        }

        // Adaptive threshold: compute local mean in 11x11 neighborhood
        // pixel < (local_mean - 20) → pupil candidate
        const halfWin = 5;
        const threshOffset = 20;
        const binary = new Uint8Array(roiW * roiH); // 1 = pupil candidate
        for (let y = halfWin; y < roiH - halfWin; y++) {
            for (let x = halfWin; x < roiW - halfWin; x++) {
                let localSum = 0;
                let count = 0;
                for (let dy = -halfWin; dy <= halfWin; dy++) {
                    for (let dx = -halfWin; dx <= halfWin; dx++) {
                        localSum += blurred[(y + dy) * roiW + (x + dx)];
                        count++;
                    }
                }
                const localMean = localSum / count;
                binary[y * roiW + x] = (blurred[y * roiW + x] < localMean - threshOffset) ? 1 : 0;
            }
        }

        // Find connected components using flood fill, pick largest
        const visited = new Uint8Array(roiW * roiH);
        let bestSize = 0;
        let bestMinX = 0, bestMaxX = 0, bestMinY = 0, bestMaxY = 0;

        for (let y = halfWin; y < roiH - halfWin; y++) {
            for (let x = halfWin; x < roiW - halfWin; x++) {
                if (binary[y * roiW + x] === 1 && !visited[y * roiW + x]) {
                    // BFS flood fill
                    const queue = [[x, y]];
                    visited[y * roiW + x] = 1;
                    let size = 0;
                    let minX = x, maxX = x, minY = y, maxY = y;

                    while (queue.length > 0) {
                        const [cx, cy] = queue.shift();
                        size++;
                        if (cx < minX) minX = cx;
                        if (cx > maxX) maxX = cx;
                        if (cy < minY) minY = cy;
                        if (cy > maxY) maxY = cy;

                        for (const [nx, ny] of [[cx-1,cy],[cx+1,cy],[cx,cy-1],[cx,cy+1]]) {
                            if (nx >= 0 && nx < roiW && ny >= 0 && ny < roiH &&
                                binary[ny * roiW + nx] === 1 && !visited[ny * roiW + nx]) {
                                visited[ny * roiW + nx] = 1;
                                queue.push([nx, ny]);
                            }
                        }
                    }

                    if (size > bestSize) {
                        bestSize = size;
                        bestMinX = minX; bestMaxX = maxX;
                        bestMinY = minY; bestMaxY = maxY;
                    }
                }
            }
        }

        // Need minimum blob size to be meaningful
        if (bestSize < 4) return null;

        // Fit bounding circle: radius = half of max(width, height)
        const blobW = bestMaxX - bestMinX + 1;
        const blobH = bestMaxY - bestMinY + 1;
        const pupilDiameter = Math.max(blobW, blobH);
        const pupilRadiusPx = pupilDiameter / 2;

        // Sanity: pupil should be smaller than iris
        if (irisR > 0 && pupilRadiusPx > irisR * 1.2) return null;
        if (irisR > 0 && pupilRadiusPx < irisR * 0.1) return null;

        const pupilIrisRatio = irisR > 0 ? (pupilRadiusPx * 2) / (irisR * 2) : null;
        return { pupilRadiusPx, pupilIrisRatio };
    }

    /**
     * Detect pupil in both eyes using canvas image processing.
     * Falls back to MediaPipe iris landmark measurement if canvas detection fails.
     * @param {HTMLCanvasElement} canvas - canvas with current video frame
     * @param {Array} landmarks - MediaPipe face landmarks
     * @param {number} videoWidth - actual video width
     * @param {number} videoHeight - actual video height
     * @returns {{pupilSize: number, pupilIrisRatio: number|null, method: string}}
     */
    function detectPupilInEye(canvas, landmarks, videoWidth, videoHeight) {
        if (!landmarks || landmarks.length < 478) {
            return { pupilSize: null, pupilIrisRatio: null, method: 'none' };
        }

        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        // Get iris radii from MediaPipe (for ratio denominator)
        const leftIrisR = irisRadiusPx(landmarks, 468, [469, 470, 471, 472]);
        const rightIrisR = irisRadiusPx(landmarks, 473, [474, 475, 476, 477]);

        // Left eye: corners 33 (inner) and 133 (outer)
        const leftResult = detectPupilInOneEye(ctx, w, h,
            landmarks[33], landmarks[133], leftIrisR);

        // Right eye: corners 362 (inner) and 263 (outer)
        const rightResult = detectPupilInOneEye(ctx, w, h,
            landmarks[362], landmarks[263], rightIrisR);

        if (leftResult && rightResult) {
            const avgRatio = (leftResult.pupilIrisRatio + rightResult.pupilIrisRatio) / 2;
            const avgRadius = (leftResult.pupilRadiusPx + rightResult.pupilRadiusPx) / 2;
            return { pupilSize: avgRatio || avgRadius, pupilIrisRatio: avgRatio, method: 'canvas_pupil' };
        } else if (leftResult) {
            return { pupilSize: leftResult.pupilIrisRatio || leftResult.pupilRadiusPx,
                     pupilIrisRatio: leftResult.pupilIrisRatio, method: 'canvas_pupil' };
        } else if (rightResult) {
            return { pupilSize: rightResult.pupilIrisRatio || rightResult.pupilRadiusPx,
                     pupilIrisRatio: rightResult.pupilIrisRatio, method: 'canvas_pupil' };
        }

        // Fallback: use old iris-boundary measurement from MediaPipe
        const irisRadius = extractPupilRadius(landmarks);
        return { pupilSize: irisRadius, pupilIrisRatio: null, method: 'landmark_iris' };
    }

    async function initFaceMesh() {
        if (faceMesh) return;
        const mpWarn = document.getElementById('mediapipe-warning');
        if (mpWarn) {
            mpWarn.textContent = 'Initializing iris tracking...';
            mpWarn.style.display = 'block';
        }
        try {
            console.log('Loading MediaPipe Vision module...');
            const vision = await import(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs'
            );

            console.log('Initializing FilesetResolver...');
            const filesetResolver = await vision.FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
            );

            console.log('Creating FaceLandmarker...');
            // Try GPU first, fall back to CPU (some mobile devices lack WebGL2 support)
            let delegate = 'GPU';
            try {
                faceMesh = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
                    baseOptions: {
                        modelAssetPath:
                            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                        delegate: 'GPU',
                    },
                    runningMode: 'VIDEO',
                    numFaces: 1,
                });
            } catch (gpuErr) {
                console.warn('GPU delegate failed, falling back to CPU:', gpuErr);
                delegate = 'CPU';
                faceMesh = await vision.FaceLandmarker.createFromOptions(filesetResolver, {
                    baseOptions: {
                        modelAssetPath:
                            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                        delegate: 'CPU',
                    },
                    runningMode: 'VIDEO',
                    numFaces: 1,
                });
            }
            console.log(`FaceLandmarker ready (${delegate}) — iris tracking active`);
            if (mpWarn) mpWarn.style.display = 'none';
        } catch (e) {
            console.warn('FaceMesh init failed, falling back to simulated data:', e);
            faceMesh = null;
            if (mpWarn) {
                mpWarn.textContent = 'Iris tracking unavailable \u2014 using estimated pupil data';
                mpWarn.style.display = 'block';
            }
        }
    }

    /**
     * Measure camera quality by running a few frames of face detection.
     * Shows a quality indicator to the user (resolution, iris size, detection method).
     */
    async function measureCameraQuality(videoEl) {
        const qualityDiv = document.getElementById('camera-quality');
        if (!qualityDiv) return;

        const resText = document.getElementById('camera-res-text');
        const irisText = document.getElementById('camera-iris-text');
        const detectText = document.getElementById('camera-detect-text');
        const qualityBadge = document.getElementById('camera-quality-badge');
        const qualityIcon = document.getElementById('camera-quality-icon');

        // Show resolution
        const resW = cameraResolution.width || videoEl.videoWidth || 0;
        const resH = cameraResolution.height || videoEl.videoHeight || 0;
        if (resText) resText.textContent = `${resW}x${resH}`;

        // Measure iris size over 10 frames
        let irisReadings = [];
        let canvasDetections = 0;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = videoEl.videoWidth || 640;
        tempCanvas.height = videoEl.videoHeight || 480;
        const tempCtx = tempCanvas.getContext('2d');

        for (let i = 0; i < 10; i++) {
            if (faceMesh && videoEl.readyState >= 2) {
                try {
                    const result = faceMesh.detectForVideo(videoEl, performance.now());
                    if (result.faceLandmarks && result.faceLandmarks.length > 0) {
                        const lm = result.faceLandmarks[0];
                        const iris = extractPupilRadius(lm);
                        if (iris) irisReadings.push(iris);

                        // Test canvas detection
                        tempCtx.drawImage(videoEl, 0, 0, tempCanvas.width, tempCanvas.height);
                        const canvasResult = detectPupilInEye(tempCanvas, lm,
                            videoEl.videoWidth, videoEl.videoHeight);
                        if (canvasResult.method === 'canvas_pupil') canvasDetections++;
                    }
                } catch (e) { /* ignore */ }
            }
            await new Promise(r => setTimeout(r, 100));
        }

        const avgIris = irisReadings.length > 0
            ? irisReadings.reduce((a, b) => a + b, 0) / irisReadings.length : 0;

        if (irisText) irisText.textContent = avgIris.toFixed(1);
        if (detectText) {
            detectText.textContent = canvasDetections > 5 ? 'Canvas Pupil' : 'Landmark Iris';
        }

        // Color-code quality
        let quality, qualityClass;
        if (avgIris > 40 && canvasDetections > 5) {
            quality = 'High';
            qualityClass = 'high';
        } else if (avgIris >= 20) {
            quality = 'Medium';
            qualityClass = 'medium';
        } else {
            quality = 'Low';
            qualityClass = 'low';
        }

        // Store quality in state for adaptive timing
        cameraQuality = qualityClass;

        if (qualityBadge) {
            qualityBadge.textContent = quality;
            qualityBadge.className = 'camera-quality-badge ' + qualityClass;
        }
        if (qualityIcon) {
            qualityIcon.textContent = qualityClass === 'high' ? '\u2705' :
                                     qualityClass === 'medium' ? '\u26A0\uFE0F' : '\u274C';
        }

        qualityDiv.style.display = 'block';

        // Show actionable advisory when quality is LOW
        let advisoryEl = document.getElementById('camera-quality-advisory');
        if (!advisoryEl) {
            advisoryEl = document.createElement('div');
            advisoryEl.id = 'camera-quality-advisory';
            advisoryEl.style.cssText = 'font-size:0.85rem;color:#c62828;margin-top:0.5rem;padding:0.5rem 1rem;background:#fce4ec;border-radius:8px;border:1px solid #e94560;';
            qualityDiv.querySelector('.camera-quality-inner').appendChild(advisoryEl);
        }
        if (qualityClass === 'low') {
            advisoryEl.textContent = 'Tip: Improve lighting, move closer to camera, or ensure your face is clearly visible for reliable results.';
            advisoryEl.style.display = 'block';
        } else {
            advisoryEl.style.display = 'none';
        }

        // Update live pupil unit label
        const unitEl = document.getElementById('live-pupil-unit');
        if (unitEl && canvasDetections > 5) {
            unitEl.textContent = 'ratio';
        }
    }

    /** Run face mesh detection; update lastLandmarks; return pupil radius.
     *  Also updates face-presence counters and blink detection. */
    function detectPupil(videoEl, phase) {
        let faceDetected = false;
        if (faceMesh && videoEl.readyState >= 2) {
            try {
                const result = faceMesh.detectForVideo(videoEl, performance.now());
                if (result.faceLandmarks && result.faceLandmarks.length > 0) {
                    lastLandmarks = result.faceLandmarks[0];
                    faceDetected = true;
                    updateBlinkDetection(lastLandmarks);

                    // Update face status indicator
                    const dot = document.getElementById('face-status-dot');
                    const txt = document.getElementById('face-status-text');
                    if (dot) { dot.className = 'face-dot detected'; }
                    if (txt) { txt.textContent = 'Face detected'; }

                    return extractPupilRadius(lastLandmarks);
                }
            } catch (e) { /* ignore frame errors */ }
        }

        // No face detected
        lastLandmarks = null;
        const dot = document.getElementById('face-status-dot');
        const txt = document.getElementById('face-status-text');
        if (dot) { dot.className = 'face-dot absent'; }
        if (txt) { txt.textContent = 'No face'; }

        return null;  // don't inject fake data
    }

    /** Wrapper that also increments face-presence counters per phase.
     *  Returns { pupilSize, pupilIrisRatio, method } when canvas detection
     *  is used during slideshow, or just a number for backward compat. */
    function detectPupilWithPresence(videoEl, phase, canvasForPupil) {
        const hasFaceMesh = faceMesh && videoEl.readyState >= 2;
        let faceDetected = false;
        let pupilRadius = null;
        let pupilIrisRatio = null;
        let detectionMethod = 'random_fallback';

        if (hasFaceMesh) {
            try {
                const result = faceMesh.detectForVideo(videoEl, performance.now());
                if (result.faceLandmarks && result.faceLandmarks.length > 0) {
                    lastLandmarks = result.faceLandmarks[0];
                    faceDetected = true;
                    updateBlinkDetection(lastLandmarks);

                    // Try canvas-based pupil detection first (the core fix)
                    if (canvasForPupil && phase === 'slideshow') {
                        const canvasResult = detectPupilInEye(
                            canvasForPupil, lastLandmarks, actualVideoWidth,
                            videoEl.videoHeight || 720);
                        if (canvasResult.method === 'canvas_pupil' && canvasResult.pupilIrisRatio) {
                            pupilRadius = canvasResult.pupilIrisRatio;
                            pupilIrisRatio = canvasResult.pupilIrisRatio;
                            detectionMethod = 'canvas_pupil';
                            canvasPupilReadings++;
                            canvasPupilActive = true;
                        } else {
                            // Canvas detection failed, fall back to iris landmarks
                            pupilRadius = extractPupilRadius(lastLandmarks);
                            detectionMethod = 'landmark_iris';
                            landmarkFallbackReadings++;
                        }
                    } else {
                        pupilRadius = extractPupilRadius(lastLandmarks);
                        detectionMethod = 'landmark_iris';
                        if (phase === 'slideshow') landmarkFallbackReadings++;
                    }
                    if (phase === 'slideshow') realIrisReadings++;
                }
            } catch (e) { /* ignore frame errors */ }
        }

        // Update face status indicator
        const dot = document.getElementById('face-status-dot');
        const txt = document.getElementById('face-status-text');
        if (faceDetected) {
            if (dot) dot.className = 'face-dot detected';
            if (txt) txt.textContent = 'Face detected';
        } else {
            lastLandmarks = null;
            if (dot) dot.className = 'face-dot absent';
            if (txt) txt.textContent = 'No face';
            pupilRadius = null;  // don't inject fake data — only real detections enter readings
            detectionMethod = 'random_fallback';
            if (phase === 'slideshow') {
                fallbackReadings++;
                randomFallbackReadings++;
            }
        }

        // Track face presence per phase
        if (phase === 'calibration') {
            calTotalFrames++;
            if (faceDetected) calFaceFrames++;
        } else if (phase === 'slideshow') {
            slideTotalFrames++;
            if (faceDetected) slideFaceFrames++;
        }

        lastDetectionMethod = detectionMethod;
        return pupilRadius;
    }

    // ─── Mobile detection ──────────────────────────────────────────────────
    const isMobile = /Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    // ─── Toast notifications ────────────────────────────────────────────────
    function showToast(message, type = 'error') {
        const container = document.getElementById('toast-container');
        if (!container) { console.warn('Toast:', message); return; }
        const toast = document.createElement('div');
        toast.className = 'toast toast-' + type;
        toast.textContent = message;
        container.appendChild(toast);
        // Trigger animation
        requestAnimationFrame(() => toast.classList.add('show'));
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // ─── API helpers ────────────────────────────────────────────────────────
    async function fetchJSON(url, opts = {}) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000);
        try {
            const resp = await fetch(url, { ...opts, signal: controller.signal });
            clearTimeout(timeoutId);
            if (!resp.ok) {
                const errBody = await resp.json().catch(() => ({}));
                throw new Error(errBody.error || `HTTP ${resp.status}`);
            }
            return resp.json();
        } catch (err) {
            clearTimeout(timeoutId);
            if (err.name === 'AbortError') {
                throw new Error('Request timed out');
            }
            throw err;
        }
    }

    // ─── Phase 1: Select ────────────────────────────────────────────────────
    async function loadSports() {
        sportsData = await fetchJSON('/api/available_sports');
        sportSelect.innerHTML = '';
        const sports = Object.keys(sportsData);
        if (sports.length === 0) {
            sportSelect.innerHTML = '<option value="">No sports available</option>';
            return;
        }
        sports.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s;
            opt.textContent = s;
            sportSelect.appendChild(opt);
        });
        updateLevels();
    }

    function updateLevels() {
        const sport = sportSelect.value;
        const levels = sportsData[sport] || [];
        levelSelect.innerHTML = '';
        levels.forEach(l => {
            const opt = document.createElement('option');
            opt.value = l;
            opt.textContent = l;
            levelSelect.appendChild(opt);
        });
        btnBegin.disabled = !sport || levels.length === 0;
    }

    sportSelect.addEventListener('change', updateLevels);

    // ─── Persona helpers ─────────────────────────────────────────────────
    function showPersonaBanner(persona) {
        const banner = document.getElementById('persona-banner');
        if (!banner || !persona) return;
        document.getElementById('persona-avatar').textContent = persona.initials;
        document.getElementById('persona-name').textContent = persona.name;
        document.getElementById('persona-role').textContent = persona.role;
        document.getElementById('persona-score').textContent = persona.base_score;
        banner.style.display = 'flex';
    }

    function updatePersonaTrustScore(persona) {
        const card = document.getElementById('persona-result-card');
        if (!card || !persona) { if (card) card.style.display = 'none'; return; }
        const isPositive = persona.score_delta > 0;
        const deltaClass = isPositive ? 'positive' : 'negative';
        const deltaSign = isPositive ? '+' : '';
        const scoreColor = isPositive ? '#2e7d32' : '#c62828';
        card.innerHTML = `
            <div class="persona-result-avatar">${persona.initials}</div>
            <div class="persona-result-info">
                <div class="persona-result-name">${persona.name}</div>
                <div class="persona-result-role">${persona.role}</div>
                <div class="persona-result-bio">${persona.bio}</div>
            </div>
            <div class="persona-result-scores">
                <div class="persona-score-before">
                    <div class="persona-score-value">${persona.base_score}</div>
                    <div class="persona-score-label">Before Test</div>
                </div>
                <div class="persona-score-arrow">
                    <span class="score-delta ${deltaClass}">${deltaSign}${persona.score_delta}</span>
                    <span class="arrow-icon">&rarr;</span>
                </div>
                <div class="persona-score-after">
                    <div class="persona-score-value" style="color:${scoreColor}">${persona.trust_score}</div>
                    <div class="persona-score-label">After Test</div>
                </div>
            </div>
        `;
        card.style.display = 'flex';
    }

    btnBegin.addEventListener('click', async () => {
        const sport = sportSelect.value;
        const level = levelSelect.value;
        if (!sport || !level) return;

        btnBegin.disabled = true;
        try {
            const resp = await fetchJSON('/start_slideshow', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sport, level }),
            });
            numImages = resp.num_images || 0;
            wrongIndices = resp.wrong_indices || [];
            if (numImages === 0) {
                showToast('No images found for this selection.', 'error');
                btnBegin.disabled = false;
                return;
            }

            // Fetch random persona for this test run
            try {
                const personaResp = await fetchJSON('/api/persona');
                currentPersona = personaResp;
                showPersonaBanner(currentPersona);
            } catch (_) { /* persona is optional */ }

            try {
                await startCamera(learnVideo);
            } catch (e) {
                btnBegin.disabled = false;
                return;
            }
            await initFaceMesh();
            // Measure camera quality (non-blocking — runs in background)
            measureCameraQuality(learnVideo);
            showPhase('learning');
            startLearningPass();
        } catch (err) {
            showToast('Failed to start assessment: ' + err.message, 'error');
            btnBegin.disabled = false;
        }
    });

    // ─── Retry learning pass (insufficient blinks) ────────────────────────
    const retryBtn = document.getElementById('btn-retry-calibration');
    if (retryBtn) {
        retryBtn.addEventListener('click', () => {
            retryBtn.style.display = 'none';
            startLearningPass();
        });
    }

    // ─── Phase 2: Learning Pass ────────────────────────────────────────────
    function startLearningPass() {
        calibrationSamples = [];
        facialBaselineSamples = [];
        lastLandmarks = null;
        learnProgress.style.width = '0%';
        learnCounter.textContent = `Studying image 1 / ${numImages}`;

        // Reset liveness state
        blinkCount = 0;
        eyesClosed = false;
        earBaseline = null;
        earBaselineBuf = [];
        earThreshold = EAR_FALLBACK_THRESHOLD;
        earLogCounter = 0;
        calFaceFrames = 0;
        calTotalFrames = 0;
        const blinkEl = document.getElementById('blink-count');
        if (blinkEl) blinkEl.textContent = '0';

        // Match canvas resolution to actual video resolution
        function syncLearnCanvas() {
            const w = learnVideo.videoWidth  || 640;
            const h = learnVideo.videoHeight || 480;
            learnCanvas.width  = w;
            learnCanvas.height = h;
        }
        learnVideo.addEventListener('loadedmetadata', syncLearnCanvas, { once: true });
        syncLearnCanvas();

        // Separate face detection from blink checking:
        // - Face detection runs in its own async loop (may be slow on mobile CPU)
        // - Blink detection runs every animation frame using cached landmarks (60fps)
        // This ensures blinks aren't missed even if face detection is slow.
        let faceDetectRunning = false;
        let learnDetectActive = true;
        async function faceDetectLoop() {
            while (learnDetectActive) {
                if (!faceDetectRunning) {
                    faceDetectRunning = true;
                    detectPupilWithPresence(learnVideo, 'calibration');
                    faceDetectRunning = false;
                }
                await new Promise(r => setTimeout(r, 80));
            }
        }
        faceDetectLoop();

        function renderLoop() {
            // Check blinks every frame using cached landmarks (fast, no ML inference)
            if (lastLandmarks) {
                updateBlinkDetection(lastLandmarks);
            }
            renderFrame(learnVideo, learnCtx, learnCanvas);
            learnRafId = requestAnimationFrame(renderLoop);
        }
        learnRafId = requestAnimationFrame(renderLoop);

        let learnIdx = 0;

        // Show first learning image
        learnImg.src = `/get_learning_image/${learnIdx}?t=${Date.now()}`;

        // Collect pupil baseline samples every 500ms + facial baseline
        const sampleInterval = setInterval(() => {
            const ps = detectPupilWithPresence(learnVideo, 'calibration');
            if (ps !== null) {
                calibrationSamples.push(ps);
            }
            if (lastLandmarks) {
                collectFacialBaselineSample(lastLandmarks);
            }
        }, CAL_SAMPLE_INTERVAL);

        // Advance images every 4s
        const imageInterval = setInterval(() => {
            learnIdx++;
            if (learnIdx < numImages) {
                learnImg.src = `/get_learning_image/${learnIdx}?t=${Date.now()}`;
                learnCounter.textContent = `Studying image ${learnIdx + 1} / ${numImages}`;
                const pct = ((learnIdx + 1) / numImages) * 100;
                learnProgress.style.width = pct + '%';
            } else {
                // Learning pass complete
                clearInterval(imageInterval);
                clearInterval(sampleInterval);
                cancelAnimationFrame(learnRafId);
                learnRafId = null;
                learnDetectActive = false;

                if (blinkCount < 1) {
                    learnCounter.textContent = `Not enough blinks detected (${blinkCount}/1). Please blink naturally and retry.`;
                    const retryBtn = document.getElementById('btn-retry-calibration');
                    if (retryBtn) retryBtn.style.display = 'inline-block';
                    return;
                }
                finalizeFacialBaseline();
                submitLearningCalibration();
            }
        }, IMAGE_DURATION_MS);

        // Set initial progress
        learnProgress.style.width = ((1 / numImages) * 100) + '%';
    }

    async function submitLearningCalibration() {
        learnCounter.textContent = 'Processing calibration...';
        const calFacePresence = calTotalFrames > 0
            ? calFaceFrames / calTotalFrames : 0;
        try {
            const calResult = await fetchJSON('/api/calibrate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    pupil_sizes: calibrationSamples,
                    blink_count: blinkCount,
                    face_presence_ratio: calFacePresence,
                }),
            });
            calibrationPupilSize = calResult.calibration_pupil_size;
            startTransition();
        } catch (err) {
            learnCounter.textContent = err.message || 'Calibration failed. Please retry.';
            const retryBtn = document.getElementById('btn-retry-calibration');
            if (retryBtn) retryBtn.style.display = 'inline-block';
        }
    }

    // ─── Phase 2b: Transition ───────────────────────────────────────────────
    function startTransition() {
        showPhase('transition');
        setTimeout(() => {
            slideVideo.srcObject = stream;
            showPhase('slideshow');
            startSlideshow();
        }, 3000);
    }

    // ─── Phase 3: Slideshow ─────────────────────────────────────────────────
    function startSlideshow() {
        currentImageIndex = 0;
        slideshowReadings = [];
        slideshowFacialReadings = [];
        lastLandmarks = null;
        imageStartTimes = {};
        slideFaceFrames = 0;
        slideTotalFrames = 0;
        realIrisReadings = 0;
        fallbackReadings = 0;
        canvasPupilReadings = 0;
        landmarkFallbackReadings = 0;
        randomFallbackReadings = 0;
        canvasPupilActive = false;

        // Size mini canvas to video resolution
        slideVideo.addEventListener('loadedmetadata', () => {
            slideCanvas.width  = slideVideo.videoWidth  || 640;
            slideCanvas.height = slideVideo.videoHeight || 480;
        }, { once: true });
        slideCanvas.width  = 640;
        slideCanvas.height = 480;

        showImage(currentImageIndex);
        startPupilTracking();
    }

    function showImage(idx) {
        slideImg.src = `/get_image/${idx}?t=${Date.now()}`;
        imageStartTimes[idx] = performance.now();
        // Reset smoothing state — each image is a new context,
        // old history would reject valid readings as outliers
        emaPupil = null;
        pupilHistory = [];
        imageCounter.textContent = `Image ${idx + 1} / ${numImages}`;
        const pct = ((idx + 1) / numImages) * 100;
        slideProgress.style.width = pct + '%';

        // Adaptive duration: give more time for lower-quality cameras to accumulate valid readings
        const adaptiveDuration = cameraQuality === 'high' ? IMAGE_DURATION_MS
                               : cameraQuality === 'medium' ? 5000
                               : 6000;  // LOW or unknown: 6 seconds per image

        setTimeout(() => {
            currentImageIndex++;
            if (currentImageIndex < numImages) {
                showImage(currentImageIndex);
            } else {
                finishSlideshow();
            }
        }, adaptiveDuration);
    }

    /** Apply EMA smoothing and outlier rejection to a raw pupil reading.
     *  Returns the smoothed value, or null if rejected as outlier. */
    function smoothPupil(raw) {
        if (raw == null) return null;

        // Outlier rejection: discard if >2 std from running average
        if (pupilHistory.length >= 5) {
            const mean = pupilHistory.reduce((a, b) => a + b, 0) / pupilHistory.length;
            const variance = pupilHistory.reduce((a, v) => a + (v - mean) ** 2, 0) / pupilHistory.length;
            const std = Math.sqrt(variance);
            if (std > 0 && Math.abs(raw - mean) > OUTLIER_STD_MULT * std) {
                return null;  // likely blink or detection glitch
            }
        }

        // EMA smoothing
        if (emaPupil === null) {
            emaPupil = raw;
        } else {
            emaPupil = EMA_ALPHA * raw + (1 - EMA_ALPHA) * emaPupil;
        }

        // Track history for std calculation
        pupilHistory.push(raw);
        if (pupilHistory.length > PUPIL_HISTORY_SIZE) {
            pupilHistory.shift();
        }

        return emaPupil;
    }

    function startPupilTracking() {
        // Reset smoothing state for new slideshow
        emaPupil = null;
        pupilHistory = [];

        // Use faster sampling when canvas pupil detection may be active
        TRACKING_INTERVAL = TRACKING_INTERVAL_FAST;

        trackingInterval = setInterval(() => {
            // Draw video frame to canvas first so we can read pixels for pupil detection
            renderFrame(slideVideo, slideCtx, slideCanvas);
            const rawPs = detectPupilWithPresence(slideVideo, 'slideshow', slideCanvas);
            if (rawPs === null) { return; }
            const ps = smoothPupil(rawPs);
            if (ps === null) return;

            // Display: show ratio if canvas detection, px otherwise
            if (canvasPupilActive) {
                livePupil.textContent = ps.toFixed(3);
            } else {
                livePupil.textContent = ps.toFixed(1);
            }

            const now = performance.now();
            const onset = imageStartTimes[currentImageIndex];
            // Skip readings during the settling window after each image
            // change — the pupil needs time to adapt to new brightness
            if (onset != null && (now - onset) < SETTLE_MS) return;
            slideshowReadings.push({
                index: currentImageIndex,
                pupil_size: ps,
                timestamp_ms: now,
                image_onset_ms: onset || null,
                measurement_type: canvasPupilActive ? 'ratio' : 'pixel',
                detection_method: lastDetectionMethod,
            });

            // Facial reaction tracking
            if (lastLandmarks && facialBaseline.eyeOpen > 0) {
                const facialScore = computeFacialReaction(lastLandmarks, facialBaseline);
                slideshowFacialReadings.push({
                    index: currentImageIndex,
                    facial_score: facialScore,
                    timestamp_ms: now,
                    image_onset_ms: onset || null,
                });
                const liveFacialEl = document.getElementById('live-facial');
                if (liveFacialEl) liveFacialEl.textContent = facialScore.toFixed(3);
            }
        }, TRACKING_INTERVAL);
    }

    function stopPupilTracking() {
        if (trackingInterval) {
            clearInterval(trackingInterval);
            trackingInterval = null;
        }
    }

    async function finishSlideshow() {
        stopPupilTracking();
        stopCamera();
        showPhase('processing');

        try {
            const slideFacePresence = slideTotalFrames > 0
                ? slideFaceFrames / slideTotalFrames : 0;
            const totalSlideReadings = realIrisReadings + fallbackReadings;
            const irisTrackingRatio = totalSlideReadings > 0
                ? realIrisReadings / totalSlideReadings : 0;

            // Determine measurement type: if majority of readings used canvas, report ratio
            const measurementType = (canvasPupilReadings > landmarkFallbackReadings) ? 'ratio' : 'pixel';

            const calFacePresence = calTotalFrames > 0
                ? calFaceFrames / calTotalFrames : 0;

            // Single stateless POST with all client-side state
            console.log('=== TRUST SCORE REQUEST ===');
            console.log('readings count:', slideshowReadings.length,
                        'calibration:', calibrationPupilSize,
                        'wrong_indices:', wrongIndices,
                        'num_images:', numImages,
                        'iris_tracking_ratio:', irisTrackingRatio,
                        'detection_stats:', { canvas_pupil: canvasPupilReadings, landmark_iris: landmarkFallbackReadings, fallback: randomFallbackReadings });
            const result = await fetchJSON('/api/trust_score', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    calibration_pupil_size: calibrationPupilSize,
                    readings: slideshowReadings,
                    wrong_indices: wrongIndices,
                    num_images: numImages,
                    blink_count: blinkCount,
                    calibration_face_presence: calFacePresence,
                    slideshow_face_presence: slideFacePresence,
                    iris_tracking_ratio: irisTrackingRatio,
                    measurement_type: measurementType,
                    detection_stats: {
                        canvas_pupil: canvasPupilReadings,
                        landmark_iris: landmarkFallbackReadings,
                        fallback: randomFallbackReadings,
                    },
                    facial_readings: slideshowFacialReadings,
                    facial_baseline: facialBaseline.eyeOpen || 0,
                    persona: currentPersona,
                }),
            });

            console.log('=== TRUST SCORE RESPONSE ===');
            console.log(JSON.stringify(result, null, 2));

            if (result.error) {
                console.error('Trust score error:', result.error);
                showToast('Error computing results: ' + result.error, 'error');
                showPhase('select');
                btnBegin.disabled = false;
                return;
            }

            setTimeout(() => showResults(result), 1500);
        } catch (err) {
            console.error('finishSlideshow error:', err);
            showToast('Assessment failed: ' + err.message, 'error');
            showPhase('select');
            btnBegin.disabled = false;
        }
    }

    // ─── Phase 5: Results ───────────────────────────────────────────────────
    function showResults(data) {
        showPhase('results');

        // Persona result card (above verdict)
        if (data.persona) {
            updatePersonaTrustScore(data.persona);
            // Update banner score to final score
            const bannerScore = document.getElementById('persona-score');
            if (bannerScore) bannerScore.textContent = data.persona.trust_score;
        } else {
            const card = document.getElementById('persona-result-card');
            if (card) card.style.display = 'none';
        }

        // Top verdict badge (PASS / FAIL / LIVENESS FAILED / INCONCLUSIVE / LOW CONFIDENCE)
        const badge = document.getElementById('verdict-badge');
        badge.textContent = data.demo_mode ? data.verdict + ' (Demo)' : data.verdict;
        if (data.verdict === 'PASS') {
            badge.className = 'verdict-badge verified';
        } else if (data.verdict === 'LOW CONFIDENCE' || data.verdict === 'INCONCLUSIVE') {
            badge.className = 'verdict-badge ' + (data.verdict === 'INCONCLUSIVE' ? 'inconclusive' : 'low-confidence');
        } else {
            badge.className = 'verdict-badge not-verified';
        }

        // Large verdict result in the grid card
        const verdictResult = document.getElementById('verdict-result');
        verdictResult.textContent = data.verdict;
        if (data.verdict === 'PASS') {
            verdictResult.className = 'verdict-result pass';
        } else if (data.verdict === 'LOW CONFIDENCE') {
            verdictResult.className = 'verdict-result low-confidence';
        } else if (data.verdict === 'INCONCLUSIVE') {
            verdictResult.className = 'verdict-result inconclusive';
        } else {
            verdictResult.className = 'verdict-result fail';
        }

        // Iris tracking warning (live mode with poor tracking) or INCONCLUSIVE reason
        const irisWarnEl = document.getElementById('iris-warning');
        if (irisWarnEl) {
            if (data.iris_warning) {
                irisWarnEl.textContent = data.iris_warning;
                irisWarnEl.style.display = 'block';
            } else if (data.inconclusive_reason) {
                irisWarnEl.textContent = data.inconclusive_reason;
                irisWarnEl.style.display = 'block';
            } else {
                irisWarnEl.style.display = 'none';
            }
        }

        // Liveness badge
        const livenessBadge = document.getElementById('liveness-badge');
        if (livenessBadge) {
            if (data.liveness_passed) {
                livenessBadge.textContent = 'Liveness: PASSED';
                livenessBadge.className = 'liveness-badge passed';
            } else {
                livenessBadge.textContent = 'Liveness: FAILED';
                livenessBadge.className = 'liveness-badge failed';
                const reasonEl = document.getElementById('liveness-reason');
                if (reasonEl) reasonEl.textContent = data.liveness_reason || '';
            }
        }

        // Detection quality badge
        const detQualBadge = document.getElementById('detection-quality-badge');
        if (detQualBadge && data.detection_quality) {
            const dq = data.detection_quality;
            detQualBadge.textContent = `Detection: ${dq}`;
            detQualBadge.className = 'detection-quality-badge ' + dq.toLowerCase();
            if (dq === 'LOW') {
                const reasonEl = document.getElementById('liveness-reason');
                if (reasonEl && !data.liveness_reason) {
                    reasonEl.textContent = 'Detection quality insufficient \u2014 result may be unreliable';
                }
            }
        }

        // Update stat labels for ratio vs pixel
        const isRatio = data.measurement_type === 'ratio';
        const unitLabel = isRatio ? '(ratio)' : '(px)';

        document.getElementById('stat-baseline').textContent =
            (data.calibration_pupil_size || 0).toFixed(1);
        document.getElementById('stat-detected').textContent =
            `${data.detected_count} / ${data.total_wrong}`;
        document.getElementById('stat-effect-size').textContent =
            data.trust_score != null ? data.trust_score.toFixed(2) : '15.00';
        document.getElementById('stat-swapped-mean').textContent =
            data.swapped_mean != null ? data.swapped_mean.toFixed(2) : '--';
        document.getElementById('stat-correct-mean').textContent =
            data.correct_mean != null ? data.correct_mean.toFixed(2) : '--';

        document.getElementById('stat-reaction-time').textContent =
            data.avg_reaction_time_ms != null ? data.avg_reaction_time_ms.toFixed(0) : 'N/A';
        document.getElementById('stat-confidence').textContent =
            data.confidence_weight != null ? data.confidence_weight.toFixed(4) : '1.0000';

        // Facial reaction stats
        const facialEffEl = document.getElementById('stat-facial-effect');
        if (facialEffEl) facialEffEl.textContent =
            data.facial_effect_size != null ? data.facial_effect_size.toFixed(4) : '0.0000';
        const compositeEl = document.getElementById('stat-composite');
        if (compositeEl) compositeEl.textContent =
            data.trust_score != null
                ? `${data.trust_score.toFixed(2)} (50% detection + 30% magnitude + 20% contrast)`
                : '--';

        // ── Trust score gauge ──
        const gaugeFill = document.getElementById('effect-gauge-fill');
        if (gaugeFill) {
            const ts = Math.max(15, Math.min(data.trust_score || 15, 100));
            const pct = ((ts - 15) / 85) * 100;
            const isPass = data.verdict === 'PASS';
            gaugeFill.style.width = '0%';
            gaugeFill.style.background = isPass
                ? 'linear-gradient(90deg, #4caf50, #2e7d32)'
                : 'linear-gradient(90deg, #e94560, #c62828)';
            // Animate after a short delay
            setTimeout(() => { gaugeFill.style.width = pct + '%'; }, 200);
        }

        // ── Group comparison chart (Correct vs Swapped means) ──
        const groupCanvas = document.getElementById('chart-group');
        if (groupCanvas) {
            new Chart(groupCanvas, {
                type: 'bar',
                data: {
                    labels: ['Correct Images (mean)', 'Swapped Images (mean)'],
                    datasets: [{
                        label: 'Mean Pupil Size (px)',
                        data: [data.correct_mean || 0, data.swapped_mean || 0],
                        backgroundColor: [chartColor('--chart-correct'), chartColor('--chart-swapped')],
                        borderColor: [chartColor('--chart-correct-border'), chartColor('--chart-swapped-border')],
                        borderWidth: 1,
                    }],
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                afterLabel() {
                                    return `Trust Score: ${(data.trust_score || 15).toFixed(2)}`;
                                }
                            }
                        }
                    },
                    scales: { y: { beginAtZero: false, title: { display: true, text: 'Pupil Size (px)' } } },
                },
                plugins: [{
                    id: 'groupBaseline',
                    afterDraw(chart) {
                        const baseline = data.calibration_pupil_size;
                        if (baseline == null) return;
                        const y = chart.scales.y.getPixelForValue(baseline);
                        const ctx = chart.ctx;
                        ctx.save();
                        ctx.setLineDash([6, 4]);
                        ctx.strokeStyle = chartColor('--chart-baseline'); ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(chart.chartArea.left, y);
                        ctx.lineTo(chart.chartArea.right, y);
                        ctx.stroke();
                        ctx.fillStyle = chartColor('--chart-baseline'); ctx.font = '11px sans-serif';
                        ctx.fillText('Baseline', chart.chartArea.right - 50, y - 6);
                        ctx.restore();
                    }
                }],
            });
        }

        // ── Bar chart ──
        const barColors = data.image_labels.map((_, i) =>
            data.wrong_indices.includes(i) ? chartColor('--chart-swapped') : chartColor('--chart-correct')
        );
        const barBorders = data.image_labels.map((_, i) =>
            data.wrong_indices.includes(i) ? chartColor('--chart-swapped-border') : chartColor('--chart-correct-border')
        );

        new Chart(document.getElementById('chart-bar'), {
            type: 'bar',
            data: {
                labels: data.image_labels,
                datasets: [{
                    label: 'Pupil Size (px)',
                    data: data.per_image_sizes,
                    backgroundColor: barColors,
                    borderColor: barBorders,
                    borderWidth: 1,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            afterLabel(ctx) {
                                const i = ctx.dataIndex;
                                if (data.wrong_indices.includes(i) && data.reaction_times && data.reaction_times[String(i)] != null) {
                                    return `Reaction: ${data.reaction_times[String(i)].toFixed(0)} ms`;
                                }
                                return '';
                            }
                        }
                    }
                },
                scales: { y: { beginAtZero: false, title: { display: true, text: 'Pupil Size (px)' } } },
            },
            plugins: [{
                id: 'baselineLine',
                afterDraw(chart) {
                    const baseline = data.calibration_pupil_size;
                    if (baseline == null) return;
                    const y = chart.scales.y.getPixelForValue(baseline);
                    const ctx = chart.ctx;
                    ctx.save();
                    ctx.setLineDash([6, 4]);
                    ctx.strokeStyle = '#4caf50'; ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.moveTo(chart.chartArea.left, y);
                    ctx.lineTo(chart.chartArea.right, y);
                    ctx.stroke();
                    ctx.fillStyle = '#4caf50'; ctx.font = '11px sans-serif';
                    ctx.fillText('Baseline', chart.chartArea.right - 50, y - 6);
                    ctx.restore();
                }
            }],
        });

        // ── Line chart ──
        const lineColors = data.dilation_percentages.map((_, i) =>
            data.wrong_indices.includes(i) ? chartColor('--chart-swapped-border') : chartColor('--chart-correct-border')
        );

        new Chart(document.getElementById('chart-line'), {
            type: 'line',
            data: {
                labels: data.image_labels,
                datasets: [{
                    label: 'Dilation %',
                    data: data.dilation_percentages,
                    borderColor: chartColor('--chart-line'),
                    backgroundColor: chartColor('--chart-line-fill'),
                    fill: true, tension: 0.3,
                    pointBackgroundColor: lineColors,
                    pointRadius: data.dilation_percentages.map((_, i) =>
                        data.wrong_indices.includes(i) ? 8 : 4
                    ),
                    pointBorderColor: lineColors,
                    pointBorderWidth: 2,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { title: { display: true, text: 'Dilation %' } } },
            },
        });

        // ── Reaction Time chart ──
        const rtLabels = [];
        const rtValues = [];
        const rtColors = [];
        if (data.reaction_times) {
            for (const idx of data.wrong_indices) {
                rtLabels.push(`Image ${idx + 1}`);
                const rt = data.reaction_times[String(idx)];
                rtValues.push(rt != null ? rt : 0);
                // Color code: green ≤ 1500ms, orange 1500-3000ms, red > 3000ms
                if (rt != null && rt <= 1500) {
                    rtColors.push('rgba(76, 175, 80, 0.8)');
                } else if (rt != null && rt <= 3000) {
                    rtColors.push('rgba(255, 152, 0, 0.8)');
                } else {
                    rtColors.push(chartColor('--chart-swapped'));
                }
            }
        }

        const reactionCanvas = document.getElementById('chart-reaction');
        if (reactionCanvas && rtLabels.length > 0) {
            new Chart(reactionCanvas, {
                type: 'bar',
                data: {
                    labels: rtLabels,
                    datasets: [{
                        label: 'Reaction Time (ms)',
                        data: rtValues,
                        backgroundColor: rtColors,
                        borderColor: rtColors.map(c => c.replace('0.8', '1')),
                        borderWidth: 1,
                    }],
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true, title: { display: true, text: 'Reaction Time (ms)' } } },
                },
            });
        }

        // ── Facial Reaction per-image chart ──
        const facialCanvas = document.getElementById('chart-facial');
        if (facialCanvas && data.has_facial_data && data.per_image_facial_scores) {
            const facialColors = data.image_labels.map((_, i) =>
                data.wrong_indices.includes(i) ? chartColor('--chart-swapped') : chartColor('--chart-facial')
            );
            const facialBorders = data.image_labels.map((_, i) =>
                data.wrong_indices.includes(i) ? chartColor('--chart-swapped-border') : chartColor('--chart-facial-border')
            );
            new Chart(facialCanvas, {
                type: 'bar',
                data: {
                    labels: data.image_labels,
                    datasets: [{
                        label: 'Facial Reaction Score',
                        data: data.per_image_facial_scores,
                        backgroundColor: facialColors,
                        borderColor: facialBorders,
                        borderWidth: 1,
                    }],
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true, max: 1, title: { display: true, text: 'Facial Score (0-1)' } } },
                },
            });
        }

        // Update group comparison chart to show both signals if facial data present
        if (data.has_facial_data) {
            const groupCanvas2 = document.getElementById('chart-group');
            if (groupCanvas2 && groupCanvas2._chart) {
                // Already rendered — skip (Chart.js doesn't support easy update here)
            }
        }
    }

    // ─── Boot ───────────────────────────────────────────────────────────────

    /** Fetch server config and sync UI (banner + toggle) */
    async function loadConfig() {
        try {
            const config = await fetchJSON('/api/config');
            syncDemoUI(config.demo_mode, config.demo_scenario);
        } catch (e) {
            console.warn('Could not load config:', e);
        }
    }

    function syncDemoUI(isDemo, scenario) {
        const banner = document.getElementById('demo-mode-banner');
        const toggle = document.getElementById('demo-toggle');
        const labelLive = document.getElementById('mode-label-live');
        const labelDemo = document.getElementById('mode-label-demo');
        const scenarioSelect = document.getElementById('demo-scenario');
        if (banner) {
            banner.style.display = isDemo ? 'block' : 'none';
            if (isDemo && scenario) {
                const label = scenario === 'pass' ? 'Expert scenario' : 'Novice scenario';
                banner.textContent = 'DEMO MODE \u2014 ' + label;
            }
        }
        if (toggle) toggle.checked = isDemo;
        if (labelLive) labelLive.classList.toggle('active', !isDemo);
        if (labelDemo) labelDemo.classList.toggle('active', isDemo);
        if (scenarioSelect) {
            scenarioSelect.style.display = isDemo ? 'inline-block' : 'none';
            if (scenario) scenarioSelect.value = scenario;
        }
    }

    // Wire up the toggle switch
    const demoToggle = document.getElementById('demo-toggle');
    if (demoToggle) {
        demoToggle.addEventListener('change', async () => {
            const newMode = demoToggle.checked;
            try {
                const config = await fetchJSON('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ demo_mode: newMode }),
                });
                syncDemoUI(config.demo_mode, config.demo_scenario);
            } catch (e) {
                console.error('Failed to toggle demo mode:', e);
                demoToggle.checked = !newMode;  // revert on failure
            }
        });
    }

    // Wire up the scenario dropdown
    const scenarioSelect = document.getElementById('demo-scenario');
    if (scenarioSelect) {
        scenarioSelect.addEventListener('change', async () => {
            try {
                const config = await fetchJSON('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ demo_scenario: scenarioSelect.value }),
                });
                syncDemoUI(config.demo_mode, config.demo_scenario);
            } catch (e) {
                console.error('Failed to set demo scenario:', e);
            }
        });
    }

    // Wire up camera retry button
    const retryCameraBtn = document.getElementById('btn-retry-camera');
    if (retryCameraBtn) {
        retryCameraBtn.addEventListener('click', () => retryCameraAccess());
    }

    loadConfig();
    loadSports();

})();
