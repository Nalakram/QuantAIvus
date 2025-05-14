# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from collections import namedtuple
from textwrap import dedent

import setuptools.command.build_py
import setuptools.command.develop
from setuptools import Command, find_packages, setup

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
SRC_DIR = os.path.join(TOP_DIR, 'tf2onnx')

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

with open(os.path.join(TOP_DIR, 'VERSION_NUMBER')) as version_file:
    VersionInfo = namedtuple('VersionInfo', ['version', 'git_version'])(
        version=version_file.read().strip(),
        git_version=git_version
    )


class create_version(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        alv2_license = '# SPDX-License-Identifier: Apache-2.0\n\n'
        with open(os.path.join(SRC_DIR, 'version.py'), 'w') as f:
            f.write(alv2_license + dedent('''
            version = '{version}'
            git_version = '{git_version}'
            '''.format(**dict(VersionInfo._asdict()))))


class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.run_command('create_version')
        setuptools.command.build_py.build_py.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        self.run_command('create_version')
        setuptools.command.develop.develop.run(self)


cmdclass = {
    'create_version': create_version,
    'build_py': build_py,
    'develop': develop,
}

setup(
    name='tf2onnx',
    version=VersionInfo.version,
    description='Tensorflow to ONNX converter',
    python_requires='>=3.7,<3.13',
    extras_require={"test": ["pytest", "pytest-cov", "graphviz", "parameterized", "pyyaml", "timeout-decorator",], },
    cmdclass=cmdclass,
    packages=find_packages(include=['tf2onnx', 'tf2onnx.*']),
    license='Apache License v2.0',
    author='ONNX',
    author_email='onnx-technical-discuss@lists.lfaidata.foundation',
    url='https://github.com/onnx/tensorflow-onnx',
    install_requires=[
        'numpy>=1.14.1',
        'onnx>=1.4.1',
        'requests',
        'six',
        'flatbuffers>=1.12',
        'protobuf>=5.26.1,<6.0dev'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ]
)
