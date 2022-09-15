from os import path
import setuptools

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='maria',
    version='0.0.11',
    description="Simulates atmospheric emission for ground-based telescopes",
    long_description=long_description,
    author="Thomas Morris",
    author_email='thomasmorris@princeton.edu',
    url='https://github.com/tomachito/maria',
    python_requires='>=3.6',
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    include_package_data=True,
    package_data={
        'maria': [ 'am.npy',
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=['astropy',
                      'healpy',
                      'matplotlib',
                      'numpy',
                      'pandas',
                      'scipy',
                      'tqdm',
                      'weathergen'],
    license='MIT',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)