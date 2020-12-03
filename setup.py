from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="rl",
    version="0.0.1",
    author="Clément Bonnet",
    author_email="clement.bonnet16@gmail.com",
    description="""A package that gathers implementations of reinforcement
    learning algorithms.""",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clement-bonnet/rl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)