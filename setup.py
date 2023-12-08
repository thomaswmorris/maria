from os import path

import setuptools

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.rst"), encoding="utf-8") as readme_file:
    readme = readme_file.read()

with open(path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line
        for line in requirements_file.read().splitlines()
        if not line.startswith("#")
    ]

setuptools.setup(
    name="maria",
    version="0.9.2",
    description="Simulate ground-based telescopes.",
    long_description=readme,
    author="Thomas W. Morris",
    author_email="thomasmorris@princeton.edu",
    url="https://www.thomaswmorris.com/maria",
    python_requires=">=3.10",
    packages=setuptools.find_packages(exclude=["docs", "tests"]),
    include_package_data=True,
    package_data={
        "maria": [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
)
