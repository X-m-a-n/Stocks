# --- setup.py ---
"""Setup script for the backtesting framework package."""

from setuptools import setup, find_packages

setup(
    name="backtesting_framework",
    version="0.1.0",
    description="A modular backtesting framework for trading strategies",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "polars>=0.19.3",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "sqlalchemy>=1.4.0",
        "mysql-connector-python>=8.0.0",
        "python-dotenv>=0.19.0"
    ],
    entry_points={
        "console_scripts": [
            "backtest=backtesting_framework.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)