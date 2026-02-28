"""
Minimal setup so local editable installs work:
    pip install -e .
This lets us do `from src.data.ingestion import ...` without sys.path hacks.
"""

from setuptools import setup, find_packages

setup(
    name="flight-delay-service",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
