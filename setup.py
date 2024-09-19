from setuptools import setup
import os

def find_requirements()->list:
    # Auto-find requirements
    lib_folder = os.path.dirname(os.path.realpath(__file__))
    requirement_path = f"{lib_folder}/requirements.txt"
    if os.path.isfile(requirement_path):
        with open(requirement_path) as f:
            install_requires = f.read().splitlines()

    return install_requires


setup(
    name="MaCh3PythonUtils",
    version="1.0.0",
    url="https://github.com/mach3-software/MaCh3-PythonUtils",
    author="Henry Wallace",
    author_email="henry.wallace@rhul.ac.uk",
    packages=["MaCh3PythonUtils"],
    install_requires=find_requirements(),
)