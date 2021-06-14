#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='project',
    version='0.0.1',
    description='AI Security Term Project',
    author='Byeonggil Jung',
    author_email='jbkcose@gmail.com',
    url='https://github.com/ByeongGil-Jung/2021-AI-Security',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)
