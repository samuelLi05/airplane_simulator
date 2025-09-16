#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
import sys

from setuptools import setup, find_packages

assert sys.version_info >= (3, 6)
with open('README.md', 'r', encoding='utf8') as readme_file:
    readme = readme_file.read()


# Gather requirements from requirements_dev.txt
install_reqs = []
requirements_path = 'requirements_dev.txt'
with open(requirements_path, 'r') as f:
    install_reqs += [
        s for s in [
            line.strip(' \n') for line in f
        ] if not s.startswith('#') and s != ''
    ]
requirements = install_reqs
setup_requirements = install_reqs
test_requirements = install_reqs

setup(
    author="ccafeccafe, Adis Delanovic, Jill Platts, Andre Beckus",
    author_email='',
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9'
    ],
    description="Airlift Challenge Simulator",
    entry_points={
        'console_scripts': [
            'airlift-demo=airlift.cli:demo',
        ],
    },
    install_requires=requirements,
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='airlift',
    name='airlift-challenge',
    packages=find_packages('.'),
    data_files=[
        ('pngs', ['airlift/envs/png/box.png', 'airlift/envs/png/green_delivered.png', 'airlift/envs/png/plane.png',
                  'airlift/envs/png/red_missed.png', 'airlift/envs/png/yellow_late.png'])],
    
    url='',
    version='1.0.0',
    zip_safe=False,
)
