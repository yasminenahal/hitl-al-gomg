import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reinvent_scoring",
    version="0.0.73",
    author="MolecularAI",
    author_email="patronov@gmail.com",
    description="Scoring functions for Reinvent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MolecularAI/reinvent-scoring.git",
    package_data={"reinvent_scoring": ["scoring/score_components/synthetic_accessibility/fpscores.pkl.gz"]},
    packages=setuptools.find_packages(exclude='unittest_reinvent'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
