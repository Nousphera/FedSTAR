"""Build FedSTAR and its native Flower application assets."""

from pathlib import Path
from setuptools import find_packages, setup

setup(
    name="fedstar", version="0.3.0",
    description="Federated Self-training for Semi-supervised Audio Recognition",
    long_description=Path("README.md").read_text() if Path("README.md").exists() else "FedSTAR audio learning.",
    long_description_content_type="text/markdown",
    license="Apache-2.0", license_files=("LICENSE",), python_requires=">=3.11,<3.14",
    install_requires=["torch==2.8.0", "flwr[simulation]==1.40.0", "numpy>=2,<3",
                      "scipy>=1.14,<2", "soundfile>=0.13,<0.14"],
    package_dir={"": "src"}, packages=find_packages("src", include=["fedstar", "fedstar.*"]),
    package_data={"fedstar": ["data_splits/*/*.txt"]},
    data_files=[("share/fedstar", ["setup.py", "LICENSE"]),
                ("share/fedstar/scripts", [str(p) for p in Path("scripts").glob("*.py")])],
    entry_points={"console_scripts": ["fedstar-experiment=fedstar.experiment:main"]},
)
