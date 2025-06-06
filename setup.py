#!/usr/bin/env python3
"""
Setup script for Basketball Shot Analysis (ShotSense) MVP
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="basketball-shot-analysis",
    version="1.0.0",
    description="AI-Powered Basketball Shooting Analysis MVP - Transform any smartphone into a professional shooting coach",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ShotSense Team",
    author_email="contact@shotsense.ai",
    url="https://github.com/shotsense/basketball-shot-analysis",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "shot-analysis-demo=mvp_demo:main",
            "shot-analysis=demo:main",
        ],
    },
    
    # Package data
    package_data={
        "basketball_shot_analysis": [
            "models/*.pt",
            "data/*.json",
            "configs/*.yaml",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    # Keywords
    keywords="basketball, computer-vision, machine-learning, sports-analytics, yolo, opencv, pose-estimation",
    
    # Project URLs
    project_urls={
        "Documentation": "https://basketball-shot-analysis.readthedocs.io/",
        "Bug Reports": "https://github.com/shotsense/basketball-shot-analysis/issues",
        "Source": "https://github.com/shotsense/basketball-shot-analysis",
        "Funding": "https://github.com/sponsors/shotsense",
    },
    
    # Zip safety
    zip_safe=False,
) 