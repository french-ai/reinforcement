"""Setup for the skift package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import setuptools

INSTALL_REQUIRES = [
    'torch',
    'torchvision',
    'gym',
    'tensorboard',
]
TEST_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov', 'codecov', 'sphinx'
]

path = os.path.abspath(os.path.dirname(__file__))

with open(os.path.abspath(path + '/README.md')) as f:
    README = f.read()

setuptools.setup(
    author="french ai team",
    name='torchforce',
    license="Apache-2.0",
    description='Reinforcement learning with pytorch ',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/french-ai/reinforcement',
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES + INSTALL_REQUIRES,
    },
    classifiers=[
        'Development Status :: Alpha',
        'License :: Apache 2.0 License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
