"""Shared pytest fixtures and helpers for the wavelet_explanation test suite."""
import sys
import os

# Ensure the package root is on sys.path regardless of how tests are invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
