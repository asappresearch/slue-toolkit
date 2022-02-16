from io import open
from setuptools import find_packages, setup

setup(
    name="slue_toolkit",
    version="0.0.1",
    author="Ankita Pasad, Suwon Shon, Felix Wu",
    author_email="TBD",
    description="SLUE Toolkit",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="speech benchmark",
    license="MIT",
    url="",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "torch>=1.0.0",
        "numpy",
        "pandas>=1.0.1",
        "requests",
        "tqdm>=4.31.1",
        "matplotlib",
        "fire",
        "editdistance",
        "soundfile",
        "sklearn"
        "transformers",
        "datasets",
        "seqeval",
    ],
    entry_points={},
    include_package_data=True,
    python_requires=">=3.7",
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
