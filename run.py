#!/usr/bin/env python3
"""
Pupil Dilation Trust Assessment — Entry Point
Run this script and open http://localhost:5000 in your browser.
"""
import sys
import os

# Ensure the trust_score package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'script', 'trust_score'))

from pupil_dilation import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')

    print(f"\n  Pupil Dilation Trust Assessment")
    print(f"  ────────────────────────────────")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
