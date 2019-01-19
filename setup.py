import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="word2vec_np",
    version="0.0.1",
    author="Sergey Mkrtchyan",
    author_email="sergey.mkrtchyan@gmail.com",
    description="word2vec CBOW model implementation in numpy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkserge/word2vec_np",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)