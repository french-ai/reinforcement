"""Setup for the skift package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

INSTALL_REQUIRES = [
    'torch',
    'torchvision',
    'gym',
]
TEST_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov',
]

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="french ai team",
    name='renforce',
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
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: Alpha',
        'License :: OSI Approved :: Apache 2.0 License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
