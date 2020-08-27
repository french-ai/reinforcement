"""Setup for the skift package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import platform

import setuptools

INSTALL_REQUIRES = [
    'torch',
    'torchvision',
    'gym',
    'tensorboard',
]

INSTALL_REQUIRES_NOTBOOK = [
    'matplotlib', 'IPython'
]

if platform.system() == "Windows":
    INSTALL_REQUIRES_NOTBOOK.append('ipykernel')
elif platform.system() == "Linux":
    INSTALL_REQUIRES_NOTBOOK.append('pyvirtualdisplay')

DEV_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov', 'codecov', 'sphinx', 'sphinx-glpi-theme',
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
    python_requires=">=3.6.1",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'dev': INSTALL_REQUIRES + INSTALL_REQUIRES_NOTBOOK + DEV_REQUIRES,
        "notebook": INSTALL_REQUIRES + INSTALL_REQUIRES_NOTBOOK
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
