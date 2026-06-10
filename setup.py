import os
from setuptools import setup, find_packages

# Path to the requirements file
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")


def _read_requirements(path):
    with open(path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]


# Read the requirements from the requirements file
if os.path.exists(requirements_path):
    install_requires = _read_requirements(requirements_path)
else:
    install_requires = []

setup(
    name="diffsynth",
    version="1.1.8",
    description="Enjoy the magic of Diffusion models!",
    author="Artiprocher",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_data={"diffsynth": ["tokenizer_configs/**/**/*.*"]},
    python_requires='>=3.6',
)
