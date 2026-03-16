"""Tests for pupil dilation trust assessment."""
import sys
import os
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'script', 'trust_score')))

import pupil_dilation as pd_module
from pupil_dilation import (app, ALLOWED_SPORTS, DILATION_THRESHOLD_PCT,
                           MIN_BLINKS_CALIBRATION, FACE_PRESENCE_THRESHOLD,
                           FALSE_POSITIVE_THRESHOLD,
                           FAST_REACTION_MS, CONFIDENCE_BOOST, CONFIDENCE_NEUTRAL,
                           MIN_PUPIL_PX, MAX_PUPIL_PX, MIN_READINGS_PER_IMAGE,
                           MIN_PUPIL_RATIO, MAX_PUPIL_RATIO,
                           DEMO_MODE,
                           IRIS_TRACKING_THRESHOLD, MIN_IMAGE_COVERAGE,
                           TRUST_PASS_THRESHOLD,
                           DETECTION_WEIGHT, MAGNITUDE_WEIGHT, CONTRAST_WEIGHT,
                           PERSONAS, TEST_SCORE_DELTA,
                           get_images_for_level, _store, _validate_liveness,
                           _compute_reaction_times, _compute_confidence_weight,
                           _compute_verdict, _compute_facial_verdict,
                           _compute_trust_response)


def _readings(index, pupil_size, count=3, **extra):
    """Helper: generate `count` readings for a single image index."""
    return [{'index': index, 'pupil_size': pupil_size, 'timestamp': index * 10 + j, **extra}
            for j in range(count)]


class PupilDilationTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        _store.clear()

    def _seed_store(self, data):
        """Set a known session ID and populate the server-side store."""
        sid = 'test-session'
        _store[sid] = data
        with self.client.session_transaction() as sess:
            sess['_sid'] = sid

    # ─── Available sports ────────────────────────────────────────────────
    def test_available_sports_returns_json(self):
        resp = self.client.get('/api/available_sports')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data, dict)
        self.assertIn('Archery', data)

    def test_available_sports_has_levels(self):
        resp = self.client.get('/api/available_sports')
        data = resp.get_json()
        levels = data.get('Archery', [])
        self.assertIn('Basic', levels)
        self.assertIn('Intermediate', levels)

    # ─── Slideshow start ─────────────────────────────────────────────────
    def test_start_slideshow_valid(self):
        resp = self.client.post('/start_slideshow',
                                data=json.dumps({'sport': 'Archery', 'level': 'Basic'}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('num_images', data)
        self.assertGreater(data['num_images'], 0)

    def test_start_slideshow_invalid_sport(self):
        resp = self.client.post('/start_slideshow',
                                data=json.dumps({'sport': 'InvalidSport', 'level': 'Basic'}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    # ─── Input validation (path traversal protection) ────────────────────
    def test_path_traversal_blocked(self):
        images = get_images_for_level('../../../etc', 'passwd')
        self.assertEqual(images, [])

    def test_whitelist_enforced(self):
        images = get_images_for_level('NotASport', 'NotALevel')
        self.assertEqual(images, [])

    # ─── Calibration ─────────────────────────────────────────────────────
    def test_api_calibrate_valid(self):
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({'pupil_sizes': [10.0, 11.0, 12.0]}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('calibration_pupil_size', data)
        self.assertAlmostEqual(data['calibration_pupil_size'], 11.0, places=1)

    def test_api_calibrate_empty(self):
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({'pupil_sizes': []}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_api_calibrate_missing_field(self):
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    # ─── MIME types ──────────────────────────────────────────────────────
    def test_get_image_mime_type(self):
        self.client.post('/start_slideshow',
                         data=json.dumps({'sport': 'Archery', 'level': 'Basic'}),
                         content_type='application/json')
        resp = self.client.get('/get_image/0')
        self.assertEqual(resp.status_code, 200)
        content_type = resp.content_type
        self.assertTrue(content_type.startswith('image/'),
                        f'Expected image/* MIME type, got {content_type}')

    # ─── Threshold-based dilation detection ──────────────────────────────
    def test_threshold_constant_exists(self):
        self.assertIsInstance(DILATION_THRESHOLD_PCT, (int, float))
        self.assertGreater(DILATION_THRESHOLD_PCT, 0)
        self.assertLess(DILATION_THRESHOLD_PCT, 1)  # should be a fraction, not pixels

    def test_pupil_recognition_with_threshold(self):
        """pupil_recognition_result should use threshold, not exact float comparison."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [2],
            'pupil_data': [
                {'index': 2, 'pupil_size': 10.3, 'timestamp': 1000},
            ],
        })
        resp = self.client.get('/pupil_recognition_result')
        data = resp.get_json()
        # 10.3 - 10.0 = 0.3, which is < 5% of 10.0 (0.5), so should be 'no'
        self.assertEqual(data['result'], 'no')

    def test_pupil_recognition_detects_large_change(self):
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [2],
            'pupil_data': [
                {'index': 2, 'pupil_size': 11.0, 'timestamp': 1000},
            ],
        })
        resp = self.client.get('/pupil_recognition_result')
        data = resp.get_json()
        # 11.0 - 10.0 = 1.0 > 5% of 10.0 (0.5), so should be 'yes'
        self.assertEqual(data['result'], 'yes')

    # ─── Submit pupil data ───────────────────────────────────────────────
    def test_submit_pupil_data(self):
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({'readings': [
                                    {'index': 0, 'pupil_size': 10.0},
                                    {'index': 1, 'pupil_size': 11.5},
                                ]}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['received'], 2)

    def test_submit_pupil_data_missing_field(self):
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    # ─── Trust score ─────────────────────────────────────────────────────
    def test_trust_score_computation(self):
        """Dilation only at swapped images, not at correct → PASS."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1, 3],
            'slideshow_images': ['a', 'b', 'c', 'd', 'e'],
            'pupil_data': (
                _readings(0, 10.1) + _readings(1, 12.5) + _readings(2, 10.2) +
                _readings(3, 12.8) + _readings(4, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })

        resp = self.client.get('/api/trust_score')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'PASS')
        self.assertEqual(data['detected_count'], 2)
        self.assertEqual(data['false_positives'], 0)
        # Hybrid trust score fields
        self.assertIn('swapped_mean', data)
        self.assertIn('correct_mean', data)
        self.assertIn('trust_score', data)
        self.assertGreaterEqual(data['trust_score'], TRUST_PASS_THRESHOLD)
        self.assertGreater(data['swapped_mean'], data['correct_mean'])

    def test_trust_score_no_calibration(self):
        resp = self.client.get('/api/trust_score')
        self.assertEqual(resp.status_code, 400)

    # ─── End-to-end: calibrate → submit → trust_score ────────────────────
    def test_full_flow_via_api(self):
        """Verify the complete browser flow works through API calls."""
        # 1. Start slideshow
        resp = self.client.post('/start_slideshow',
                                data=json.dumps({'sport': 'Archery', 'level': 'Basic'}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        num_images = resp.get_json()['num_images']

        # 2. Calibrate (with liveness data)
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({
                                    'pupil_sizes': [10.0] * 30,
                                    'blink_count': 3,
                                    'face_presence_ratio': 0.95,
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)

        # 3. Submit readings (3 per image, with face presence)
        readings = []
        for i in range(num_images):
            ps = 10.0 + (1.0 if i % 3 == 0 else 0)
            readings.extend(_readings(i, ps))
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({
                                    'readings': readings,
                                    'face_presence_ratio': 0.90,
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)

        # 4. Get trust score
        resp = self.client.get('/api/trust_score')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('verdict', data)
        self.assertIn(data['verdict'], ['PASS', 'FAIL'])
        self.assertIn('per_image_sizes', data)
        self.assertEqual(len(data['per_image_sizes']), num_images)

    # ─── Index page ──────────────────────────────────────────────────────
    def test_index_page_serves_html(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'Pupil Dilation', resp.data)

    # ─── detect_pupil wrong_index bug fix ────────────────────────────────
    def test_detect_pupil_wrong_index_is_list(self):
        """Bug 2: detect_pupil should check `index in list`, not `index == list`."""
        self._seed_store({
            'slideshow_images': ['img0', 'img1', 'img2'],
            'wrong_index': [1, 2],
            'pupil_data': [],
            'wrong_image_pupil': None,
        })
        resp = self.client.post('/detect_pupil', data={'index': '1'})
        self.assertEqual(resp.status_code, 400)  # Missing frame file


    # ─── Liveness / anti-spoofing tests ─────────────────────────────────

    def test_liveness_constants_exist(self):
        """MIN_BLINKS_CALIBRATION and FACE_PRESENCE_THRESHOLD must be defined."""
        self.assertIsInstance(MIN_BLINKS_CALIBRATION, int)
        self.assertGreaterEqual(MIN_BLINKS_CALIBRATION, 1)
        self.assertIsInstance(FACE_PRESENCE_THRESHOLD, float)
        self.assertGreater(FACE_PRESENCE_THRESHOLD, 0)
        self.assertLessEqual(FACE_PRESENCE_THRESHOLD, 1)

    def test_calibrate_accepts_liveness_data(self):
        """blink_count and face_presence_ratio should be stored."""
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({
                                    'pupil_sizes': [10.0, 11.0, 12.0],
                                    'blink_count': 4,
                                    'face_presence_ratio': 0.92,
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        # Verify stored in the server-side store
        with self.client.session_transaction() as sess:
            sid = sess.get('_sid')
        store = _store[sid]
        self.assertEqual(store['blink_count'], 4)
        self.assertAlmostEqual(store['calibration_face_presence'], 0.92)

    def test_submit_pupil_data_accepts_face_presence(self):
        """face_presence_ratio should be stored from submit_pupil_data."""
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({
                                    'readings': [{'index': 0, 'pupil_size': 10.0}],
                                    'face_presence_ratio': 0.88,
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        with self.client.session_transaction() as sess:
            sid = sess.get('_sid')
        store = _store[sid]
        self.assertAlmostEqual(store['slideshow_face_presence'], 0.88)

    def test_liveness_passes_with_good_data(self):
        """blinks>=2 and presence>=80% → trust score computed normally."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['liveness_passed'])
        self.assertNotEqual(data['verdict'], 'LIVENESS FAILED')

    def test_liveness_fails_insufficient_blinks(self):
        """blinks=0 → verdict 'LIVENESS FAILED'."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'blink_count': 0,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'LIVENESS FAILED')
        self.assertFalse(data['liveness_passed'])
        self.assertIn('blinks', data['liveness_reason'])

    def test_liveness_fails_low_calibration_presence(self):
        """calibration face presence 50% → fails."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': _readings(0, 10.0),
            'blink_count': 5,
            'calibration_face_presence': 0.50,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'LIVENESS FAILED')
        self.assertFalse(data['liveness_passed'])
        self.assertIn('calibration', data['liveness_reason'])

    def test_liveness_fails_low_slideshow_presence(self):
        """slideshow face presence 50% → fails."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': _readings(0, 10.0),
            'blink_count': 5,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.50,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'LIVENESS FAILED')
        self.assertFalse(data['liveness_passed'])
        self.assertIn('slideshow', data['liveness_reason'])

    # ─── Learning pass / original_images ────────────────────────────────

    def test_start_slideshow_stores_original_images(self):
        """start_slideshow should store original_images (un-swapped) in session."""
        resp = self.client.post('/start_slideshow',
                                data=json.dumps({'sport': 'Archery', 'level': 'Basic'}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        with self.client.session_transaction() as sess:
            sid = sess.get('_sid')
        store = _store[sid]
        self.assertIn('original_images', store)
        self.assertIn('slideshow_images', store)
        # original_images should be sorted (un-swapped)
        self.assertEqual(store['original_images'], sorted(store['original_images']))
        # slideshow_images should have the same images (but potentially swapped)
        self.assertEqual(sorted(store['original_images']), sorted(store['slideshow_images']))

    def test_get_learning_image_valid(self):
        """get_learning_image should serve the correct image from original order."""
        self.client.post('/start_slideshow',
                         data=json.dumps({'sport': 'Archery', 'level': 'Basic'}),
                         content_type='application/json')
        resp = self.client.get('/get_learning_image/0')
        self.assertEqual(resp.status_code, 200)
        content_type = resp.content_type
        self.assertTrue(content_type.startswith('image/'),
                        f'Expected image/* MIME type, got {content_type}')

    def test_get_learning_image_invalid_index(self):
        """get_learning_image should return 400 for out-of-range index."""
        self.client.post('/start_slideshow',
                         data=json.dumps({'sport': 'Archery', 'level': 'Basic'}),
                         content_type='application/json')
        resp = self.client.get('/get_learning_image/999')
        self.assertEqual(resp.status_code, 400)

    def test_full_flow_with_liveness(self):
        """End-to-end flow including liveness data → normal score."""
        # 1. Start slideshow
        resp = self.client.post('/start_slideshow',
                                data=json.dumps({'sport': 'Archery', 'level': 'Basic'}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        num_images = resp.get_json()['num_images']

        # 2. Calibrate with liveness
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({
                                    'pupil_sizes': [10.0] * 30,
                                    'blink_count': 4,
                                    'face_presence_ratio': 0.92,
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)

        # 3. Submit readings (3+ per image) with face presence
        readings = []
        for i in range(num_images):
            readings.extend(_readings(i, 10.0))
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({
                                    'readings': readings,
                                    'face_presence_ratio': 0.88,
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)

        # 4. Trust score should pass liveness
        resp = self.client.get('/api/trust_score')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['liveness_passed'])
        self.assertIn('verdict', data)
        self.assertIn(data['verdict'], ['PASS', 'FAIL'])


    # ─── _compute_reaction_times tests ──────────────────────────────────

    def test_reaction_times_fast_detection(self):
        """Readings with quick dilation should produce a small reaction time."""
        pupil_data = [
            {'index': 2, 'pupil_size': 10.0, 'timestamp_ms': 100, 'image_onset_ms': 0},
            {'index': 2, 'pupil_size': 11.5, 'timestamp_ms': 500, 'image_onset_ms': 0},
        ]
        rt = _compute_reaction_times(pupil_data, [2], 10.0, 0.05)
        self.assertIn(2, rt)
        self.assertEqual(rt[2], 500)

    def test_reaction_times_no_dilation(self):
        """Readings below threshold should produce no reaction time."""
        pupil_data = [
            {'index': 2, 'pupil_size': 10.0, 'timestamp_ms': 100, 'image_onset_ms': 0},
            {'index': 2, 'pupil_size': 10.1, 'timestamp_ms': 500, 'image_onset_ms': 0},
        ]
        rt = _compute_reaction_times(pupil_data, [2], 10.0, 0.05)
        self.assertEqual(rt, {})

    def test_reaction_times_missing_timestamps(self):
        """Readings without timestamps should be gracefully skipped."""
        pupil_data = [
            {'index': 2, 'pupil_size': 12.0, 'timestamp': 1000},
        ]
        rt = _compute_reaction_times(pupil_data, [2], 10.0, 0.05)
        self.assertEqual(rt, {})

    def test_reaction_times_multiple_wrong_images(self):
        """Reaction times computed independently per swapped image."""
        pupil_data = [
            {'index': 1, 'pupil_size': 11.5, 'timestamp_ms': 300, 'image_onset_ms': 0},
            {'index': 3, 'pupil_size': 11.5, 'timestamp_ms': 800, 'image_onset_ms': 200},
        ]
        rt = _compute_reaction_times(pupil_data, [1, 3], 10.0, 0.05)
        self.assertEqual(rt[1], 300)
        self.assertEqual(rt[3], 600)

    # ─── _compute_confidence_weight tests ────────────────────────────────

    def test_confidence_weight_instant(self):
        """0ms reaction → full CONFIDENCE_BOOST."""
        w = _compute_confidence_weight({2: 0})
        self.assertAlmostEqual(w, CONFIDENCE_BOOST)

    def test_confidence_weight_fast(self):
        """Reaction at half FAST_REACTION_MS → midpoint between BOOST and NEUTRAL."""
        w = _compute_confidence_weight({2: FAST_REACTION_MS / 2})
        expected = (CONFIDENCE_BOOST + CONFIDENCE_NEUTRAL) / 2
        self.assertAlmostEqual(w, expected, places=4)

    def test_confidence_weight_at_threshold(self):
        """Reaction exactly at FAST_REACTION_MS → CONFIDENCE_NEUTRAL."""
        w = _compute_confidence_weight({2: FAST_REACTION_MS})
        self.assertAlmostEqual(w, CONFIDENCE_NEUTRAL)

    def test_confidence_weight_slow(self):
        """Reaction slower than threshold → CONFIDENCE_NEUTRAL."""
        w = _compute_confidence_weight({2: 5000})
        self.assertAlmostEqual(w, CONFIDENCE_NEUTRAL)

    def test_confidence_weight_empty(self):
        """No reaction times → CONFIDENCE_NEUTRAL."""
        w = _compute_confidence_weight({})
        self.assertAlmostEqual(w, CONFIDENCE_NEUTRAL)

    def test_confidence_weight_none_input(self):
        """None-like empty input → CONFIDENCE_NEUTRAL."""
        w = _compute_confidence_weight({})
        self.assertAlmostEqual(w, CONFIDENCE_NEUTRAL)

    def test_confidence_weight_boundary(self):
        """Reaction at exactly 0ms → CONFIDENCE_BOOST."""
        w = _compute_confidence_weight({0: 0.0})
        self.assertAlmostEqual(w, CONFIDENCE_BOOST)

    # ─── Integration: trust score with reaction time data ────────────────

    def test_trust_score_with_reaction_times(self):
        """Trust score should include reaction_times and confidence_weight."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1, 3],
            'slideshow_images': ['a', 'b', 'c', 'd', 'e'],
            'pupil_data': (
                _readings(0, 10.1) +
                _readings(1, 12.5, timestamp_ms=500, image_onset_ms=0) +
                _readings(2, 10.2) +
                _readings(3, 12.8, timestamp_ms=300, image_onset_ms=0) +
                _readings(4, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertIn('reaction_times', data)
        self.assertIn('confidence_weight', data)
        self.assertIn('avg_reaction_time_ms', data)
        self.assertGreater(data['confidence_weight'], 1.0)
        self.assertLessEqual(data['confidence_weight'], CONFIDENCE_BOOST)
        self.assertEqual(data['verdict'], 'PASS')

    def test_trust_score_backward_compat_no_timestamps(self):
        """Without timestamps, confidence_weight should be 1.0 (neutral)."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1, 3],
            'slideshow_images': ['a', 'b', 'c', 'd', 'e'],
            'pupil_data': (
                _readings(0, 10.1) + _readings(1, 12.5) + _readings(2, 10.2) +
                _readings(3, 12.8) + _readings(4, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['confidence_weight'], 1.0)
        self.assertIsNone(data['avg_reaction_time_ms'])
        self.assertEqual(data['verdict'], 'PASS')

    def test_no_dilation_at_swapped_images_is_fail(self):
        """No meaningful separation between swapped and correct → FAIL."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1, 3],
            'slideshow_images': ['a', 'b', 'c', 'd', 'e'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 10.1) + _readings(2, 10.2) +
                _readings(3, 10.1) + _readings(4, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'FAIL')
        self.assertLess(data['trust_score'], TRUST_PASS_THRESHOLD)

    def test_uniform_dilation_everywhere(self):
        """Same pupil size at ALL images → detection + magnitude high, contrast 0.
        With the hybrid formula this still scores high because dilation occurred
        on swapped images (even though it also occurred everywhere else)."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1, 3],
            'slideshow_images': ['a', 'b', 'c', 'd', 'e'],
            'pupil_data': (
                _readings(0, 12.0) + _readings(1, 12.0) + _readings(2, 12.0) +
                _readings(3, 12.0) + _readings(4, 12.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['contrast_c'], 0.0)  # no group separation

    def test_false_positive_fields_in_response(self):
        """Response should include false_positives, total_correct, false_positive_ratio."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertIn('false_positives', data)
        self.assertIn('total_correct', data)
        self.assertIn('false_positive_ratio', data)
        self.assertEqual(data['false_positives'], 0)
        self.assertEqual(data['total_correct'], 2)

    def test_liveness_failed_has_reaction_fields(self):
        """Liveness-failed response should include placeholder reaction fields."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': _readings(0, 10.0),
            'blink_count': 0,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'LIVENESS FAILED')
        self.assertIn('reaction_times', data)
        self.assertIn('confidence_weight', data)
        self.assertIn('avg_reaction_time_ms', data)
        self.assertEqual(data['confidence_weight'], 1.0)
        self.assertIsNone(data['avg_reaction_time_ms'])
        # Trust score fields present with defaults
        self.assertIn('swapped_mean', data)
        self.assertIn('correct_mean', data)
        self.assertIn('trust_score', data)
        self.assertEqual(data['trust_score'], 15.0)


    # ─── _compute_verdict tests ─────────────────────────────────────────

    def test_compute_verdict_clear_separation_pass(self):
        """Swapped images clearly higher → PASS with high trust score."""
        per_image_avg = {0: 10.0, 1: 12.5, 2: 10.1, 3: 12.8, 4: 10.0}
        verdict, sw_mean, co_mean, ts, dr, mc, cc = _compute_verdict(
            per_image_avg, [1, 3], 5, 10.0)
        self.assertEqual(verdict, 'PASS')
        self.assertGreaterEqual(ts, TRUST_PASS_THRESHOLD)
        self.assertGreater(sw_mean, co_mean)

    def test_compute_verdict_no_separation_fail(self):
        """Swapped and correct have similar readings → FAIL."""
        per_image_avg = {0: 10.0, 1: 10.1, 2: 10.2, 3: 10.1, 4: 10.0}
        verdict, sw_mean, co_mean, ts, dr, mc, cc = _compute_verdict(
            per_image_avg, [1, 3], 5, 10.0)
        self.assertEqual(verdict, 'FAIL')
        self.assertLess(ts, TRUST_PASS_THRESHOLD)

    def test_compute_verdict_equal_readings_fail(self):
        """All images have identical readings at baseline → no dilation → FAIL."""
        per_image_avg = {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0}
        verdict, sw_mean, co_mean, ts, dr, mc, cc = _compute_verdict(
            per_image_avg, [1, 3], 5, 10.0)
        self.assertEqual(verdict, 'FAIL')
        self.assertEqual(ts, 15.0)  # minimum score
        self.assertAlmostEqual(sw_mean, co_mean)

    def test_compute_verdict_single_swapped_image(self):
        """Edge case: only one swapped image."""
        per_image_avg = {0: 10.0, 1: 13.0, 2: 10.0}
        verdict, sw_mean, co_mean, ts, dr, mc, cc = _compute_verdict(
            per_image_avg, [1], 3, 10.0)
        self.assertEqual(verdict, 'PASS')
        self.assertGreaterEqual(ts, TRUST_PASS_THRESHOLD)

    def test_compute_verdict_swapped_lower_fail(self):
        """Swapped images LOWER than correct → no dilation → FAIL."""
        per_image_avg = {0: 12.0, 1: 10.0, 2: 12.0, 3: 10.0, 4: 12.0}
        verdict, sw_mean, co_mean, ts, dr, mc, cc = _compute_verdict(
            per_image_avg, [1, 3], 5, 10.0)
        self.assertEqual(verdict, 'FAIL')

    def test_compute_verdict_missing_indices(self):
        """Missing data for some images → uses available data."""
        per_image_avg = {0: 10.0, 1: 13.0}  # images 2,3,4 missing
        verdict, sw_mean, co_mean, ts, dr, mc, cc = _compute_verdict(
            per_image_avg, [1], 5, 10.0)
        # Only 2 data points but still computable
        self.assertIsInstance(verdict, str)
        self.assertIn(verdict, ['PASS', 'FAIL'])

    # ─── /api/config endpoint ────────────────────────────────────────────

    def test_api_config_returns_settings(self):
        """/api/config should return demo_mode and trust_pass_threshold."""
        resp = self.client.get('/api/config')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('demo_mode', data)
        self.assertIn('trust_pass_threshold', data)
        self.assertIsInstance(data['demo_mode'], bool)
        self.assertEqual(data['trust_pass_threshold'], TRUST_PASS_THRESHOLD)

    # ─── demo_mode in trust score response ───────────────────────────────

    def test_trust_score_includes_demo_mode(self):
        """Normal trust score response should include demo_mode field."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertIn('demo_mode', data)
        self.assertEqual(data['demo_mode'], DEMO_MODE)

    def test_liveness_failed_includes_demo_mode(self):
        """Liveness-failed response should include demo_mode field."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': _readings(0, 10.0),
            'blink_count': 0,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'LIVENESS FAILED')
        self.assertIn('demo_mode', data)
        self.assertEqual(data['demo_mode'], DEMO_MODE)

    # ─── Anatomical sanity validation ────────────────────────────────────

    def test_calibrate_rejects_tiny_pupil(self):
        """Pupil size below MIN_PUPIL_PX should be rejected."""
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({'pupil_sizes': [1.0, 1.2, 0.8]}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('too small', data['error'])

    def test_calibrate_rejects_huge_pupil(self):
        """Pupil size above MAX_PUPIL_PX should be rejected."""
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({'pupil_sizes': [25.0, 30.0, 28.0]}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('too large', data['error'])

    def test_calibrate_accepts_normal_pupil(self):
        """Pupil size within range should be accepted."""
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({'pupil_sizes': [8.0, 9.0, 10.0]}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)

    # ─── Minimum readings per image ──────────────────────────────────────

    def test_insufficient_readings_rejected(self):
        """Images with fewer than MIN_READINGS_PER_IMAGE readings should be excluded."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': [
                # Only 1 reading each — all below minimum
                {'index': 0, 'pupil_size': 10.0, 'timestamp': 1},
                {'index': 1, 'pupil_size': 12.0, 'timestamp': 2},
                {'index': 2, 'pupil_size': 10.0, 'timestamp': 3},
            ],
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('Insufficient', data['error'])

    def test_min_readings_constant(self):
        """MIN_READINGS_PER_IMAGE should be defined and >= 1."""
        self.assertIsInstance(MIN_READINGS_PER_IMAGE, int)
        self.assertGreaterEqual(MIN_READINGS_PER_IMAGE, 1)

    def test_anatomical_constants_exist(self):
        """MIN_PUPIL_PX and MAX_PUPIL_PX should be defined."""
        self.assertIsInstance(MIN_PUPIL_PX, (int, float))
        self.assertIsInstance(MAX_PUPIL_PX, (int, float))
        self.assertGreater(MAX_PUPIL_PX, MIN_PUPIL_PX)

    # ─── Ratio-based measurement tests ───────────────────────────────────

    def test_ratio_constants_exist(self):
        """MIN_PUPIL_RATIO and MAX_PUPIL_RATIO should be defined."""
        self.assertIsInstance(MIN_PUPIL_RATIO, (int, float))
        self.assertIsInstance(MAX_PUPIL_RATIO, (int, float))
        self.assertGreater(MAX_PUPIL_RATIO, MIN_PUPIL_RATIO)
        self.assertGreater(MIN_PUPIL_RATIO, 0)
        self.assertLess(MAX_PUPIL_RATIO, 1)

    def test_calibrate_ratio_valid(self):
        """Ratio-based calibration within range should succeed."""
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({
                                    'pupil_sizes': [0.40, 0.42, 0.41],
                                    'measurement_type': 'ratio',
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertAlmostEqual(data['calibration_pupil_size'], 0.41, places=1)

    def test_calibrate_ratio_too_small(self):
        """Ratio below MIN_PUPIL_RATIO should be rejected."""
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({
                                    'pupil_sizes': [0.05, 0.06, 0.04],
                                    'measurement_type': 'ratio',
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('too small', data['error'])

    def test_calibrate_ratio_too_large(self):
        """Ratio above MAX_PUPIL_RATIO should be rejected."""
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({
                                    'pupil_sizes': [0.90, 0.92, 0.91],
                                    'measurement_type': 'ratio',
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('too large', data['error'])

    def test_calibrate_pixel_still_works(self):
        """Pixel-based calibration (default) should still work as before."""
        resp = self.client.post('/api/calibrate',
                                data=json.dumps({
                                    'pupil_sizes': [8.0, 9.0, 10.0],
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)

    def test_trust_score_with_ratio_measurements(self):
        """Ratio-based readings should produce valid trust score."""
        self._seed_store({
            'calibration_pupil_size': 0.40,
            'wrong_index': [1, 3],
            'slideshow_images': ['a', 'b', 'c', 'd', 'e'],
            'pupil_data': (
                _readings(0, 0.40) + _readings(1, 0.52) + _readings(2, 0.41) +
                _readings(3, 0.54) + _readings(4, 0.40)
            ),
            'measurement_type': 'ratio',
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'PASS')
        self.assertEqual(data['measurement_type'], 'ratio')
        self.assertGreaterEqual(data['trust_score'], TRUST_PASS_THRESHOLD)

    def test_submit_pupil_data_stores_measurement_type(self):
        """measurement_type should be stored from submit_pupil_data."""
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({
                                    'readings': [{'index': 0, 'pupil_size': 0.40}],
                                    'measurement_type': 'ratio',
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        with self.client.session_transaction() as sess:
            sid = sess.get('_sid')
        store = _store[sid]
        self.assertEqual(store['measurement_type'], 'ratio')

    def test_submit_pupil_data_stores_detection_stats(self):
        """detection_stats should be stored from submit_pupil_data."""
        stats = {'canvas_pupil': 50, 'landmark_iris': 10, 'fallback': 5}
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({
                                    'readings': [{'index': 0, 'pupil_size': 10.0}],
                                    'detection_stats': stats,
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        with self.client.session_transaction() as sess:
            sid = sess.get('_sid')
        store = _store[sid]
        self.assertEqual(store['detection_stats'], stats)

    # ─── Detection quality tests ─────────────────────────────────────────

    def test_detection_quality_high(self):
        """HIGH detection quality when >70% canvas pupil readings."""
        self._seed_store({
            'calibration_pupil_size': 0.40,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 0.40) + _readings(1, 0.52) + _readings(2, 0.41)
            ),
            'measurement_type': 'ratio',
            'detection_stats': {'canvas_pupil': 80, 'landmark_iris': 10, 'fallback': 5},
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['detection_quality'], 'HIGH')

    def test_detection_quality_medium(self):
        """MEDIUM detection quality when >50% real but <70% canvas."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'detection_stats': {'canvas_pupil': 20, 'landmark_iris': 40, 'fallback': 20},
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['detection_quality'], 'MEDIUM')

    def test_detection_quality_low(self):
        """LOW detection quality when <50% real readings."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'detection_stats': {'canvas_pupil': 5, 'landmark_iris': 10, 'fallback': 60},
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['detection_quality'], 'LOW')

    def test_detection_quality_no_stats(self):
        """No detection stats → LOW quality."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['detection_quality'], 'LOW')

    def test_trust_score_includes_detection_fields(self):
        """Trust score response should include measurement_type and detection_quality."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertIn('measurement_type', data)
        self.assertIn('detection_quality', data)
        self.assertIn('detection_stats', data)
        self.assertIn(data['detection_quality'], ['HIGH', 'MEDIUM', 'LOW'])

    def test_liveness_failed_includes_detection_fields(self):
        """Liveness-failed response should include detection fields."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': _readings(0, 10.0),
            'blink_count': 0,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'LIVENESS FAILED')
        self.assertIn('measurement_type', data)
        self.assertIn('detection_quality', data)
        self.assertEqual(data['detection_quality'], 'LOW')

    # ─── Facial reaction tests ────────────────────────────────────────────

    def test_compute_facial_verdict_clear_separation(self):
        """High facial scores at swapped images → positive effect size."""
        # 5 images, indices 1 and 3 are swapped
        per_image_facial = {0: 0.05, 1: 0.50, 2: 0.04, 3: 0.48, 4: 0.06}
        effect_size, swapped_mean, correct_mean = _compute_facial_verdict(
            per_image_facial, [1, 3], 5)
        self.assertGreater(effect_size, 1.0)
        self.assertGreater(swapped_mean, correct_mean)

    def test_compute_facial_verdict_no_separation(self):
        """Mixed facial scores with no clear swapped-vs-correct pattern → low effect."""
        # Swapped indices 1,3 get 0.10 and 0.08; correct 0,2,4 get 0.11, 0.09, 0.10
        # Both groups average ~0.09-0.10 with similar spread
        per_image_facial = {0: 0.11, 1: 0.10, 2: 0.09, 3: 0.08, 4: 0.10}
        effect_size, swapped_mean, correct_mean = _compute_facial_verdict(
            per_image_facial, [1, 3], 5)
        self.assertLess(abs(effect_size), 1.0)

    def test_compute_facial_verdict_all_equal(self):
        """Identical scores → zero effect size (zero std)."""
        per_image_facial = {0: 0.10, 1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10}
        effect_size, swapped_mean, correct_mean = _compute_facial_verdict(
            per_image_facial, [1, 3], 5)
        self.assertEqual(effect_size, 0.0)

    def test_submit_pupil_data_stores_facial_readings(self):
        """Facial readings should be stored from submit_pupil_data."""
        facial = [{'index': 0, 'facial_score': 0.05, 'timestamp_ms': 100}]
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({
                                    'readings': [{'index': 0, 'pupil_size': 10.0}],
                                    'facial_readings': facial,
                                    'facial_baseline': 0.03,
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        with self.client.session_transaction() as sess:
            sid = sess.get('_sid')
        store = _store[sid]
        self.assertEqual(store['facial_data'], facial)
        self.assertAlmostEqual(store['facial_baseline'], 0.03)

    def test_trust_score_includes_facial_fields(self):
        """Trust score response should include all facial fields."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'facial_data': [
                {'index': 0, 'facial_score': 0.04},
                {'index': 1, 'facial_score': 0.45},
                {'index': 2, 'facial_score': 0.05},
            ],
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertIn('facial_effect_size', data)
        self.assertIn('facial_swapped_mean', data)
        self.assertIn('facial_correct_mean', data)
        self.assertIn('per_image_facial_scores', data)
        self.assertIn('has_facial_data', data)
        self.assertTrue(data['has_facial_data'])

    def test_trust_score_formula(self):
        """Trust score = (detection*0.5 + magnitude*0.3 + contrast*0.2) * 85 + 15."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'facial_data': [
                {'index': 0, 'facial_score': 0.04},
                {'index': 1, 'facial_score': 0.45},
                {'index': 2, 'facial_score': 0.05},
            ],
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        expected = round(
            (data['detection_ratio'] * DETECTION_WEIGHT +
             data['magnitude_c'] * MAGNITUDE_WEIGHT +
             data['contrast_c'] * CONTRAST_WEIGHT) * 85 + 15, 2)
        self.assertAlmostEqual(data['trust_score'], expected, places=1)

    def test_trust_score_no_facial_still_works(self):
        """No facial data → verdict driven purely by hybrid pupil trust score."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertFalse(data['has_facial_data'])
        self.assertEqual(data['facial_effect_size'], 0.0)
        self.assertGreaterEqual(data['trust_score'], TRUST_PASS_THRESHOLD)

    def test_trust_score_pass_with_both_signals(self):
        """Strong pupil + strong facial → PASS."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 12.0) + _readings(2, 10.0)
            ),
            'facial_data': [
                {'index': 0, 'facial_score': 0.03},
                {'index': 1, 'facial_score': 0.50},
                {'index': 2, 'facial_score': 0.04},
            ],
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'PASS')
        self.assertGreaterEqual(data['trust_score'], TRUST_PASS_THRESHOLD)

    def test_liveness_failed_includes_score_fields(self):
        """Liveness-failed response should include trust score default fields."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': _readings(0, 10.0),
            'blink_count': 0,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'LIVENESS FAILED')
        self.assertEqual(data['trust_score'], 15.0)
        self.assertEqual(data['per_image_facial_scores'], [])
        self.assertFalse(data['has_facial_data'])

    def test_hybrid_weights_sum_to_one(self):
        """Hybrid blend weights should sum to 1.0."""
        self.assertAlmostEqual(DETECTION_WEIGHT + MAGNITUDE_WEIGHT + CONTRAST_WEIGHT, 1.0)

    def test_config_includes_hybrid_fields(self):
        """Config endpoint should include hybrid weight fields."""
        resp = self.client.get('/api/config')
        data = resp.get_json()
        self.assertIn('detection_weight', data)
        self.assertIn('magnitude_weight', data)
        self.assertIn('contrast_weight', data)
        self.assertIn('trust_pass_threshold', data)

    def test_submit_pupil_data_no_facial_backward_compat(self):
        """Omitting facial data in submit should not break."""
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({
                                    'readings': [{'index': 0, 'pupil_size': 10.0}],
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        with self.client.session_transaction() as sess:
            sid = sess.get('_sid')
        store = _store[sid]
        self.assertNotIn('facial_data', store)


    # ─── Persona profile tests ────────────────────────────────────────────

    def test_persona_endpoint_returns_valid_persona(self):
        """GET /api/persona returns a dict with all required fields."""
        resp = self.client.get('/api/persona')
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        for field in ('id', 'name', 'role', 'base_score', 'bio', 'initials'):
            self.assertIn(field, data, f'Missing field: {field}')
        self.assertIn(data['id'], [p['id'] for p in PERSONAS])

    def test_persona_endpoint_stores_in_session(self):
        """Fetching a persona persists it in the server-side store."""
        resp = self.client.get('/api/persona')
        data = json.loads(resp.data)
        with self.client.session_transaction() as sess:
            sid = sess.get('_sid')
        store = _store[sid]
        self.assertIn('persona', store)
        self.assertEqual(store['persona']['id'], data['id'])

    def test_persona_scores_match_spec(self):
        """All 7 personas exist with the correct base scores."""
        expected = {1: 380, 2: 275, 3: 415, 4: 390, 5: 345, 6: 300, 7: 360}
        self.assertEqual(len(PERSONAS), 7)
        for p in PERSONAS:
            self.assertEqual(p['base_score'], expected[p['id']])

    def test_trust_score_includes_persona_pass(self):
        """PASS verdict includes persona with +40 delta."""
        persona = PERSONAS[0]
        self._seed_store({
            'calibration_pupil_size': 0.45,
            'measurement_type': 'ratio',
            'wrong_index': [2, 5],
            'slideshow_images': ['img'] * 8,
            'pupil_data': (
                _readings(0, 0.45) + _readings(1, 0.45) +
                _readings(2, 0.60) + _readings(3, 0.45) +
                _readings(4, 0.45) + _readings(5, 0.60) +
                _readings(6, 0.45) + _readings(7, 0.45)
            ),
            'blink_count': 5,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.95,
            'persona': persona,
        })
        resp = self.client.get('/api/trust_score')
        data = json.loads(resp.data)
        self.assertEqual(data['verdict'], 'PASS')
        self.assertIn('persona', data)
        self.assertEqual(data['persona']['trust_score'], persona['base_score'] + TEST_SCORE_DELTA)
        self.assertEqual(data['persona']['score_delta'], TEST_SCORE_DELTA)

    def test_trust_score_includes_persona_fail(self):
        """FAIL verdict includes persona with -40 delta."""
        persona = PERSONAS[1]
        self._seed_store({
            'calibration_pupil_size': 0.45,
            'measurement_type': 'ratio',
            'wrong_index': [2, 5],
            'slideshow_images': ['img'] * 8,
            'pupil_data': (
                _readings(0, 0.45) + _readings(1, 0.45) +
                _readings(2, 0.45) + _readings(3, 0.45) +
                _readings(4, 0.45) + _readings(5, 0.45) +
                _readings(6, 0.45) + _readings(7, 0.45)
            ),
            'blink_count': 5,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.95,
            'persona': persona,
        })
        resp = self.client.get('/api/trust_score')
        data = json.loads(resp.data)
        self.assertEqual(data['verdict'], 'FAIL')
        self.assertIn('persona', data)
        self.assertEqual(data['persona']['trust_score'], persona['base_score'] - TEST_SCORE_DELTA)
        self.assertEqual(data['persona']['score_delta'], -TEST_SCORE_DELTA)

    def test_trust_score_no_persona_backward_compat(self):
        """No persona in store → no persona key in response (backward compat)."""
        self._seed_store({
            'calibration_pupil_size': 0.45,
            'measurement_type': 'ratio',
            'wrong_index': [2, 5],
            'slideshow_images': ['img'] * 8,
            'pupil_data': (
                _readings(0, 0.45) + _readings(1, 0.45) +
                _readings(2, 0.45) + _readings(3, 0.45) +
                _readings(4, 0.45) + _readings(5, 0.45) +
                _readings(6, 0.45) + _readings(7, 0.45)
            ),
            'blink_count': 5,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.95,
        })
        resp = self.client.get('/api/trust_score')
        data = json.loads(resp.data)
        self.assertNotIn('persona', data)

    def test_persona_test_score_delta_constant(self):
        """TEST_SCORE_DELTA is exactly 40."""
        self.assertEqual(TEST_SCORE_DELTA, 40)

    # ─── New quality gate tests ──────────────────────────────────────────

    def test_iris_tracking_threshold_raised(self):
        """IRIS_TRACKING_THRESHOLD should be 0.70 (raised from 0.50)."""
        self.assertEqual(IRIS_TRACKING_THRESHOLD, 0.70)

    def test_min_image_coverage_constant(self):
        """MIN_IMAGE_COVERAGE should be 0.5."""
        self.assertEqual(MIN_IMAGE_COVERAGE, 0.5)

    def test_fallback_readings_filtered(self):
        """Readings with detection_method='random_fallback' should be excluded from analysis."""
        # Mix of real and fallback readings — fallback should be filtered out
        real_readings = [
            {'index': 0, 'pupil_size': 10.1, 'timestamp': 1, 'detection_method': 'canvas_pupil'},
            {'index': 0, 'pupil_size': 10.2, 'timestamp': 2, 'detection_method': 'canvas_pupil'},
            {'index': 0, 'pupil_size': 10.0, 'timestamp': 3, 'detection_method': 'canvas_pupil'},
            {'index': 1, 'pupil_size': 12.5, 'timestamp': 4, 'detection_method': 'canvas_pupil'},
            {'index': 1, 'pupil_size': 12.6, 'timestamp': 5, 'detection_method': 'canvas_pupil'},
            {'index': 1, 'pupil_size': 12.4, 'timestamp': 6, 'detection_method': 'canvas_pupil'},
            {'index': 2, 'pupil_size': 10.0, 'timestamp': 7, 'detection_method': 'canvas_pupil'},
            {'index': 2, 'pupil_size': 10.1, 'timestamp': 8, 'detection_method': 'canvas_pupil'},
            {'index': 2, 'pupil_size': 10.2, 'timestamp': 9, 'detection_method': 'canvas_pupil'},
        ]
        fallback_readings = [
            {'index': 0, 'pupil_size': 99.0, 'timestamp': 10, 'detection_method': 'random_fallback'},
            {'index': 1, 'pupil_size': 99.0, 'timestamp': 11, 'detection_method': 'random_fallback'},
            {'index': 2, 'pupil_size': 99.0, 'timestamp': 12, 'detection_method': 'random_fallback'},
        ]
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': real_readings + fallback_readings,
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        # Fallback readings with pupil_size=99 should be filtered;
        # result should reflect real data showing dilation at image 1
        self.assertEqual(data['verdict'], 'PASS')
        self.assertGreaterEqual(data['trust_score'], TRUST_PASS_THRESHOLD)

    def test_inconclusive_verdict_low_coverage(self):
        """INCONCLUSIVE verdict when fewer than half the images have enough readings."""
        # 5 images but only 2 have enough readings (< 50% coverage)
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1, 3],
            'slideshow_images': ['a', 'b', 'c', 'd', 'e'],
            'pupil_data': (
                _readings(0, 10.1) +  # image 0: 3 readings — OK
                _readings(1, 12.5) +  # image 1: 3 readings — OK
                # images 2, 3, 4: no readings → insufficient coverage
                []
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'INCONCLUSIVE')
        self.assertIn('inconclusive_reason', data)
        self.assertIn('coverage', data['inconclusive_reason'].lower())

    def test_inconclusive_not_triggered_with_good_coverage(self):
        """No INCONCLUSIVE when most images have enough readings."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.1) + _readings(1, 12.5) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertNotEqual(data['verdict'], 'INCONCLUSIVE')

    def test_quality_gate_on_fail_verdict(self):
        """Iris tracking gate should apply to FAIL verdict too, not just PASS."""
        # Data that would produce FAIL (no dilation at swapped images)
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 10.0) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
            'iris_tracking_ratio': 0.40,  # below IRIS_TRACKING_THRESHOLD
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        # FAIL should be overridden to LOW CONFIDENCE due to poor tracking
        self.assertEqual(data['verdict'], 'LOW CONFIDENCE')
        self.assertIsNotNone(data['iris_warning'])

    def test_quality_gate_threshold_at_boundary(self):
        """Iris tracking ratio just below threshold → LOW CONFIDENCE."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.1) + _readings(1, 12.5) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
            'iris_tracking_ratio': 0.69,  # just below 0.70
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'LOW CONFIDENCE')

    def test_quality_gate_above_threshold_pass(self):
        """Iris tracking ratio at or above threshold → normal verdict."""
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1],
            'slideshow_images': ['a', 'b', 'c'],
            'pupil_data': (
                _readings(0, 10.1) + _readings(1, 12.5) + _readings(2, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
            'iris_tracking_ratio': 0.70,  # exactly at threshold
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'PASS')

    def test_submit_preserves_detection_method(self):
        """submit_pupil_data should preserve detection_method from readings."""
        self._seed_store({})
        resp = self.client.post('/api/submit_pupil_data',
                                data=json.dumps({
                                    'readings': [
                                        {'index': 0, 'pupil_size': 10.0,
                                         'detection_method': 'canvas_pupil'},
                                        {'index': 1, 'pupil_size': 11.0,
                                         'detection_method': 'random_fallback'},
                                    ],
                                    'face_presence_ratio': 0.9,
                                }),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        sid = None
        with self.client.session_transaction() as sess:
            sid = sess.get('_sid')
        stored_data = _store[sid]['pupil_data']
        self.assertEqual(stored_data[0]['detection_method'], 'canvas_pupil')
        self.assertEqual(stored_data[1]['detection_method'], 'random_fallback')


    # ─── Stateless POST /api/trust_score tests ─────────────────────────

    def _stateless_payload(self, verdict_type='pass'):
        """Build a full stateless POST payload."""
        calibration = 10.0
        wrong_indices = [1, 3]
        num_images = 5
        if verdict_type == 'pass':
            readings = (
                _readings(0, 10.1) + _readings(1, 12.5) + _readings(2, 10.2) +
                _readings(3, 12.8) + _readings(4, 10.0)
            )
        else:
            readings = (
                _readings(0, 10.0) + _readings(1, 10.0) + _readings(2, 10.0) +
                _readings(3, 10.0) + _readings(4, 10.0)
            )
        return {
            'calibration_pupil_size': calibration,
            'readings': readings,
            'wrong_indices': wrong_indices,
            'num_images': num_images,
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
            'iris_tracking_ratio': 0.85,
            'measurement_type': 'pixel',
            'detection_stats': {'canvas_pupil': 50, 'landmark_iris': 10, 'fallback': 5},
            'facial_readings': [],
            'facial_baseline': 0.0,
            'persona': PERSONAS[0],
        }

    def test_stateless_post_trust_score_pass(self):
        """POST /api/trust_score with PASS-worthy data returns PASS verdict."""
        payload = self._stateless_payload('pass')
        resp = self.client.post('/api/trust_score',
                                data=json.dumps(payload),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'PASS')
        self.assertGreaterEqual(data['trust_score'], TRUST_PASS_THRESHOLD)
        self.assertIn('persona', data)
        self.assertEqual(data['persona']['score_delta'], TEST_SCORE_DELTA)

    def test_stateless_post_trust_score_fail(self):
        """POST /api/trust_score with FAIL-worthy data returns FAIL verdict."""
        payload = self._stateless_payload('fail')
        resp = self.client.post('/api/trust_score',
                                data=json.dumps(payload),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'FAIL')
        self.assertLess(data['trust_score'], TRUST_PASS_THRESHOLD)

    def test_stateless_post_missing_fields_returns_400(self):
        """POST /api/trust_score with missing required fields returns 400."""
        resp = self.client.post('/api/trust_score',
                                data=json.dumps({'calibration_pupil_size': 10.0}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('error', data)
        self.assertIn('Missing required fields', data['error'])

    def test_stateless_post_invalid_types_returns_400(self):
        """POST /api/trust_score with invalid field types returns 400."""
        payload = self._stateless_payload('pass')
        payload['calibration_pupil_size'] = 'not_a_number'
        resp = self.client.post('/api/trust_score',
                                data=json.dumps(payload),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('error', data)

    def test_stateless_post_empty_body_returns_400(self):
        """POST /api/trust_score with no body returns 400."""
        resp = self.client.post('/api/trust_score',
                                data='',
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_stateless_post_no_readings_returns_400(self):
        """POST /api/trust_score with empty readings returns 400."""
        payload = self._stateless_payload('pass')
        payload['readings'] = []
        resp = self.client.post('/api/trust_score',
                                data=json.dumps(payload),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    # ─── Demo scenario tests ─────────────────────────────────────────────

    def test_demo_pass_scenario_produces_pass(self):
        """Demo mode with 'pass' scenario should produce PASS verdict."""
        # Enable demo mode
        self.client.post('/api/config',
                         data=json.dumps({'demo_mode': True, 'demo_scenario': 'pass'}),
                         content_type='application/json')
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1, 3],
            'slideshow_images': ['a', 'b', 'c', 'd', 'e'],
            'pupil_data': (
                _readings(0, 10.0) + _readings(1, 10.0) + _readings(2, 10.0) +
                _readings(3, 10.0) + _readings(4, 10.0)
            ),
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'PASS')
        self.assertTrue(data['demo_mode'])
        # Restore
        self.client.post('/api/config',
                         data=json.dumps({'demo_mode': False}),
                         content_type='application/json')

    def test_demo_fail_scenario_produces_fail(self):
        """Demo mode with 'fail' scenario should produce FAIL verdict."""
        import random as _rng
        _rng.seed(12345)
        self.client.post('/api/config',
                         data=json.dumps({'demo_mode': True, 'demo_scenario': 'fail'}),
                         content_type='application/json')
        # Use 12 images so random noise averages out reliably
        imgs = [chr(ord('a') + i) for i in range(12)]
        readings = []
        for i in range(12):
            readings += _readings(i, 10.0)
        self._seed_store({
            'calibration_pupil_size': 10.0,
            'wrong_index': [1, 3],
            'slideshow_images': imgs,
            'pupil_data': readings,
            'blink_count': 3,
            'calibration_face_presence': 0.95,
            'slideshow_face_presence': 0.90,
        })
        resp = self.client.get('/api/trust_score')
        data = resp.get_json()
        self.assertEqual(data['verdict'], 'FAIL')
        self.assertTrue(data['demo_mode'])
        # Restore
        self.client.post('/api/config',
                         data=json.dumps({'demo_mode': False}),
                         content_type='application/json')

    def test_demo_config_toggle(self):
        """POST /api/config should toggle demo mode and scenario."""
        resp = self.client.post('/api/config',
                                data=json.dumps({'demo_mode': True, 'demo_scenario': 'fail'}),
                                content_type='application/json')
        data = resp.get_json()
        self.assertTrue(data['demo_mode'])
        self.assertEqual(data['demo_scenario'], 'fail')

        resp = self.client.post('/api/config',
                                data=json.dumps({'demo_mode': False}),
                                content_type='application/json')
        data = resp.get_json()
        self.assertFalse(data['demo_mode'])

    # ─── Input validation tests ──────────────────────────────────────────

    def test_detect_pupil_bad_index_returns_400(self):
        """detect_pupil with non-numeric index should return 400."""
        resp = self.client.post('/detect_pupil',
                                data={'index': 'abc'})
        self.assertEqual(resp.status_code, 400)

    def test_calibrate_missing_body_returns_400(self):
        """Calibrate with no JSON body returns 400."""
        resp = self.client.post('/api/calibrate',
                                content_type='application/json')
        self.assertEqual(resp.status_code, 400)

    def test_start_slideshow_returns_wrong_indices(self):
        """start_slideshow response should include wrong_indices."""
        resp = self.client.post('/start_slideshow',
                                data=json.dumps({'sport': 'Archery', 'level': 'Basic'}),
                                content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('wrong_indices', data)
        self.assertEqual(len(data['wrong_indices']), 2)


if __name__ == '__main__':
    unittest.main()
