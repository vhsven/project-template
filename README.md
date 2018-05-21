# vhsven-sklearn - My scikit-learn extensions

This python package contains an extension class for scikit-learn's
decision tree algorithms. Where scikit-learn by default uses early
stopping criteria to prevent a tree from growing too large,
this package contains a `PruneableDecisionTreeClassifier` which
uses one of two well-known pruning techniques to keep the size
in check. The two techniques are Reduced Error Pruning and
Error Based Pruning.

Created based on **project-template**, a template project for
[scikit-learn](http://scikit-learn.org/) compatible extensions.
See also: http://contrib.scikit-learn.org/project-template/.

## Installation and Usage
The package by itself comes with a single module and a classifier. Before
installing the module you will need `numpy` and `scipy`.
To install the module execute:
```shell
$ python setup.py install
```
or 
```
pip install project-template
```
or (when using Anaconda)
```
conda develop /path/to/project
```

If the installation is successful, and `scikit-learn` is correctly installed,
you should be able to execute the following in Python:
```python
>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import cross_val_score
>>> from pruneabletree import PruneableDecisionTreeClassifier
>>> clf = PruneableDecisionTreeClassifier(random_state=0, prune='rep')
>>> iris = load_iris()
>>> cross_val_score(clf, iris.data, iris.target, cv=10)
```

Developers will need to install additional packages to contribute to the code.
When using Anaconda, create a new environment based on the `environment_dev.yml`
file. Else, take a look at the `Makefile` to see which packages were originally
installed.

## Documentation

The documentation is built using [sphinx](http://www.sphinx-doc.org/en/stable/).
It incorporates narrative documentation from the `doc/` directory, standalone
examples from the `examples/` directory, and API reference compiled from
estimator docstrings.

The online documentation can be found [here](https://vhsven.github.io/vhsven-sklearn/).

To build the documentation locally, ensure that you have `sphinx`,
`sphinx-gallery` and `matplotlib` by executing:
```shell
$ pip install sphinx matplotlib sphinx-gallery
```
The documentation contains a home page (`doc/index.rst`), an API
documentation page (`doc/api.rst`) and a page documenting the `template` module 
(`doc/template.rst`). Sphinx allows you to automatically document your modules
and classes by using the `autodoc` directive (see `template.rst`). To change the
aesthetics of the docs and other parameters, edit the `doc/conf.py` file. For
more information visit the [Sphinx Documentation](http://www.sphinx-doc.org/en/stable/contents.html).

You can also add code examples in the `examples` folder. All files inside
the folder of the form `plot_*.py` will be executed and their generated
plots will be available for viewing in the `/auto_examples` URL.

To build the documentation locally execute
```shell
$ cd doc
$ make html
```

The project uses [CircleCI](https://circleci.com/) to build its documentation
from the `master` branch and host it using [Github Pages](https://pages.github.com/).

## Uploading your package to PyPI

Uploading your package to [PyPI](https://pypi.python.org/pypi) allows users to
install your package through `pip`. Python provides two repositories to upload
your packages. The [PyPI Test](https://testpypi.python.org/pypi) repository,
which is to be used for testing packages before their release, and the
[PyPI](https://pypi.python.org/pypi) repository, where you can make your
releases. You need to register a username and password with both these sites.
The username and passwords for both these sites need not be the same. To upload
your package through the command line, you need to store your username and
password in a file called `.pypirc` in your `$HOME` directory with the
following format.

```shell
[distutils]
index-servers =
  pypi
  pypitest

[pypi]
repository=https://pypi.python.org/pypi
username=<your-pypi-username>
password=<your-pypi-passowrd>

[pypitest]
repository=https://testpypi.python.org/pypi
username=<your-pypitest-username>
password=<your-pypitest-passowrd>
```
Make sure that all details in `setup.py` are up to date. To upload your package
to the Test server, execute:
```
python setup.py register -r pypitest
python setup.py sdist upload -r pypitest
```
Your package should now be visible on: https://testpypi.python.org/pypi

To install a package from the test server, execute:
```
pip install -i https://testpypi.python.org/pypi <package-name>
```

Similarly, to upload your package to the PyPI server execute
```
python setup.py register -r pypi
python setup.py sdist upload -r pypi
```
To install your package, execute:
```
pip install <package-name>
```
