from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hitl_al_gomg",
    version="0.0.2",
    license="MIT",
    author="Yasmine Nahal",
    author_email="yasmine.nahal@aalto.fi",
    description="A Human-in-the-loop active learning workflow to improve molecular property predictors with human expert feedback for goal-oriented molecule generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yasminenahal/hitl_al_gomg.git",
    download_url = 'https://github.com/yasminenahal/hitl_al_gomg/archive/v_02.tar.gz',
    keywords = ['REINVENT', 'HITL', 'HITL_AL_GOMG'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)