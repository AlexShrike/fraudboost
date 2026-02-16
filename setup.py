"""
Setup script for FraudBoost library.
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fraudboost",
    version="0.1.0",
    author="Alex Shrike",
    author_email="alex@example.com",  # Update with actual email
    description="A gradient boosting framework purpose-built for fraud detection in fintech",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexShrike/fraudboost",
    project_urls={
        "Bug Tracker": "https://github.com/AlexShrike/fraudboost/issues",
        "Documentation": "https://github.com/AlexShrike/fraudboost#readme",
        "Source Code": "https://github.com/AlexShrike/fraudboost",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "examples": [
            "xgboost>=1.6.0",
            "jupyter",
            "seaborn",
        ],
        "full": [
            "pytest>=6.0",
            "pytest-cov",
            "xgboost>=1.6.0",
            "jupyter",
            "seaborn",
        ]
    },
    keywords="machine-learning, fraud-detection, gradient-boosting, fintech, xgboost, anomaly-detection",
    include_package_data=True,
    zip_safe=False,
)