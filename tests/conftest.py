"""Shared test fixtures."""
import os
import sys

# Make ``src/`` importable when tests are run without an editable install.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
