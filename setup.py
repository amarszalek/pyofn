from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ['numpy', 'scipy', 'pandas', 'matplotlib', 'setuptools']


setup(
    name="pyofn",
    version="0.0.10",
    author="Adam Marszałek",
    author_email="amarszalek@pk.edu.pl",
    description="Python package for Ordered Fuzzy Numbers",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/amarszalek/pyofn",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
    ],
)