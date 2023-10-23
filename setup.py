import os
import re
from setuptools import setup, find_packages

def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'subtitlr', '__version__.py')
    pattern = r'^__version__ = "(.+)"'
    with open(version_file, 'r') as f:
        lines = f.read()
    match = re.search(pattern, lines, re.M)
    if match:
        print(match.group(1))
        return match.group(1)
    else:
        return '0.0.0'


setup(
    name='subtitlr',
    version=get_version(),
    description='Automatic subtitle generation and translation',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    url='https://github.com/adrian-valente/subtitlr/',
    author='Adrian Valente',
    license='GPLv3',
    entry_points={
        'console_scripts': [
            'subtitlr = subtitlr.__main__:main'
        ],
    },
    install_requires=[
        'pyyaml',
        'requests',
        'tqdm',
        'googletrans==3.1.0a0',
        'faster-whisper==0.9.0',  
    ],
)
    