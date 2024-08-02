# Introduction
This package contains the scoring functions for REINVENT.

# Installation
To install in REINVENT's environment either install from repo or use `pip install reinvent-scoring` for the latest
official release.

# Developing
## Setup environment
You can use Conda to create an environment with all the necessary packages installed.

```
$ conda env create -f reinvent_scoring
[...]
$ conda activate reinvent_scoring
```

## Run tests
The tests use the `unittest` package testing framework.  Before you can run the tests make sure that you have created a
`config.json`file in the `reinvent_scoring/configs` directory.  There is an example config in the same directory, which 
you can base your own config off of.  Make sure that you set `MAIN_TEST_PATH` to a non-existent directory; it is where 
temporary files will be written during the tests; if it is set to an existing directory, that directory will be removed 
once the tests have finished.

Some tests require a proprietary OpenEye license; you have to set up a few things to make the tests read your
license.  The simple way is to just set the `OE_LICENSE` environment variable to the path of the file containing the
license.  If you just want to set the license in the `reinvent_scoring` Conda environment, it is a bit more complicated,
but you only have to do it once.

```
(reinvent-scoring) $ cd $CONDA_PREFIX
$ mkdir -p etc/conda/activate.d
$ mkdir -p etc/conda/deactivate.d
```

Put the following in `etc/conda/activate.d/env_vars.sh`.

```
#!/bin/sh
export OE_LICENSE='</path/to/your/oe_license/file>'
```

And put the following in `etc/conda/deactivate.d/env_vars.sh`.

```
#!/bin/sh
unset OE_LICENSE
```

Once you have created the files, deactivate and re-activate the environment, and `echo $OE_LICENSE` should output the
path to the license file.

Once you have created and configured your environment, you can run unittests by running

```bash
python main_test.py --unittests
```

If you have a valid Open eye license and other dependencie configured, like Icolos and AZDOCK - 
you can also run integration tests, by running command (remember to submit this configuration, since the default one is test):

```bash
python main_test.py --integration --base_config <path to your configuration>
```

# Building
- Building: `python setup.py sdist bdist_wheel`
- Upload build to test: `python -m twine upload --repository testpypi dist/*`
- Upload build: `python -m twine upload dist/*`

