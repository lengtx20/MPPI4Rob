#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mppi4rob",
    version="0.1.0",
    author="lengtx20",
    description="A lightweight MPPI controller library for robotics using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lengtx20/MPPI4Rob",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "gymnasium>=0.28.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "matplotlib>=3.3.0",
        ],
        "examples": [
            "matplotlib>=3.3.0",
            "gymnasium>=0.28.0",
        ],
    },
)
