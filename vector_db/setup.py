from setuptools import setup, find_packages

setup(
    name="vector_db",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "sentence-transformers"
    ],
)
