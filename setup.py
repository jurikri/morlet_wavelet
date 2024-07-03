
from setuptools import setup, find_packages

setup(
    name='msanalysis',
    version='0.1',
    packages=['morlet_wavelet'],
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'ray',
    ],
    author='jurikri',
    author_email='jurikri@gmail..com',
    description='A package to perform signal analysis using Morlet wavelet transforms on EEG data',
    url='https://github.com/jurikri/morlet_wavelet',
)
