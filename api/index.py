import sys
import os

# Add the trust_score directory to the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'script', 'trust_score'))

from pupil_dilation import app

# Vercel expects the WSGI app as `app`
