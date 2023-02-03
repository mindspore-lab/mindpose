#!/usr/bin/env python

from setuptools import find_packages, setup

exec(open("mindpose/version.py").read())

install_requires = open("requirements.txt").read().strip().split("\n")

setup(
    name="mindpose",
    author="MindSpore Ecosystem",
    author_email="mindspore-ecosystem@example.com",
    url="https://github.com/mindspore-lab/mindpose",
    project_urls={
        "Sources": "https://github.com/mindspore-lab/mindpose",
        "Issue Tracker": "https://github.com/mindspore-lab/mindpose/issues",
    },
    description="Aan open-source toolbox for pose estimation based on MindSpore.",
    license="Apache Software License 2.0",
    include_package_data=True,
    packages=find_packages(include=["mindpose", "mindpose.*"]),
    setup_requires=[
        "setuptools >= 18.0",
    ],
    install_requires=install_requires,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    test_suite="tests",
    tests_require=[
        "pytest",
    ],
    version=__version__,
    zip_safe=False,
)
