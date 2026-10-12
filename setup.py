import os
from pathlib import Path

import setuptools

script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

config_root = Path("navsim/planning/script/config")
hydra_config_files = [
    str(path.relative_to(config_root))
    for path in sorted(config_root.rglob("*.yaml"))
]

setuptools.setup(
    name="maneuverflow-navsim-v1",
    version="1.0.0",
    author="LatentPilot",
    description="ManeuverFlow-TF core training, inference, and evaluation for NAVSIM v1",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/wyh-autonomy/maneuverflow",
    python_requires=">=3.9",
    # Include NAVSIM namespace directories and packaged Hydra configurations.
    packages=setuptools.find_namespace_packages(
        where=script_folder,
        include=("navsim", "navsim.*"),
        exclude=("*.tests", "*.tests.*"),
    ),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        "navsim.planning.script.config": hydra_config_files,
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
