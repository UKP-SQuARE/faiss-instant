from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="faiss-instant",
    version="0.0.1",
    author="Kexin Wang",
    author_email="kexin.wang.2049@gmail.com",
    description="This package contains toolkit for faiss-instant. It mainly helps to encode texts via Transformers and build Faiss indexes in an automatic way.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/UKP-SQuARE/faiss-instant",
    project_urls={
        "Bug Tracker": "https://github.com/UKP-SQuARE/faiss-instant/issues",
    },
    packages=find_packages('faiss_instant'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        'beir',
        'crash-ipdb',
        'sentence-transformers',
        'faiss-gpu'
    ],
)