"""
Setup script for MarsRippleNet
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="marsripplenet",
    version="1.0.0",
    author="MarsRipple Research Team",
    description="Mars-RippleNet: Spatial Adaptive Dual-Branch Architecture for Martian Aeolian Ripple Segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/",
    packages=find_packages(exclude=["tests", "scripts"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "segmentation-models-pytorch>=0.3.3",
        "timm>=0.9.0",
        "albumentations>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.2.0",
        "pandas>=1.5.0",
        "Pillow>=9.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="mars segmentation deep-learning pytorch remote-sensing",
)
