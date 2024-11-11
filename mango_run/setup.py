from setuptools import setup, find_packages

setup(
    name="mango_run",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "importlib",
    ],
)
