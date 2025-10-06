# tests/conftest.py
# Ensure the package root (week_04) is on sys.path when running pytest.
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
