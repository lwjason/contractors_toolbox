import pathlib

import pkg_resources
from setuptools import setup

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='toolbox',
    version='',
    packages=['toolbox'],
    url='',
    license='',
    author='Pohan',
    author_email='l.w.jasons@gmail.com',
    description='Contractors toolbox',
    install_requires=install_requires,
)