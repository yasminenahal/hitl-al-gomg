from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="hitl_al_gomg",
    version="0.0.30",
    license="MIT",
    author="Yasmine Nahal",
    author_email="yasmine.nahal@aalto.fi",
    description="A Human-in-the-loop active learning workflow to improve molecular property predictors with human expert feedback for goal-oriented molecule generation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yasminenahal/hitl_al_gomg.git",
    keywords=["REINVENT", "HITL", "HITL_AL_GOMG"],
    packages=find_packages(),
    package_data={"hitl_al_gomg": ["models/priors/random.prior.new", "scoring/chemspace/chembl.csv"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <3.11",
    install_requires=[
        "PyTDC==1.0.7",
        "scipy==1.10.1",
        "torch==1.12.1",
        "fcd-torch==1.0.7",
        "click==8.1.7",
        "matplotlib==3.9.2",
        "jupyter==1.1.1",
        "PySide2>5.15",
        "cairosvg",
        "pydantic",
        "pyyaml",
        "rdeditor==0.2.0.1",
    ],
)
