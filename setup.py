from setuptools import setup, find_packages

setup(
    name="pistachio",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "pistachio=pistachio.__main__:main",
        ],
    },
    author="Esra Büşra Işık",
    description="PISTACHIO: Proteomics-guided Integration and Spatial Transcriptomics Analysis using Constrained Hierarchical Inference and Optimization.",
    long_description="A Python package for integrating spatial transcriptomics and proteomics data using constrained NMF-based deconvolution.",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pistachio",
    license="MIT",
)