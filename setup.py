import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="poet_estimator",
    version="0.1",
    author="Dizhou Wu",
    author_email="dizhouwubruce@outlook.com",
    description="Python Implementation for Large Covariance Estimation by Thresholding Principal Orthogonal Complements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brucewuquant/POET",
    packages=setuptools.find_packages(),
    install_requires=['scipy>=1.5.1', 'numpy>=1.19.0'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
)