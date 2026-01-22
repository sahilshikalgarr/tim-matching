from setuptools import setup, find_packages

# Read the contents of README file (with error handling)
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Two-Stage Interpretable Matching for Causal Inference"

setup(
    name="tim-matching",
    version="0.1.0",
    author="Sahil Shikalgar",
    author_email="shikalgar.s@northeastern.edu",
    description="Two-Stage Interpretable Matching for Causal Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sahilshikalgarr/tim-matching",
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    keywords="causal-inference matching statistics observational-studies",
    project_urls={
        "Bug Reports": "https://github.com/sahilshikalgarr/tim-matching/issues",
        "Source": "https://github.com/sahilshikalgarr/tim-matching",
    },
)
