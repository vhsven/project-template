# vhsven-sklearn - My scikit-learn extensions

This python 3.6 package contains an extension class for scikit-learn's
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
python setup.py install
```

or

```shell
pip install pruneabletree
```

or (when using Anaconda)

```shell
conda develop /path/to/vhsven-sklearn
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
pip install sphinx matplotlib sphinx-gallery
```

You can also add code examples in the `examples` folder. All files inside
the folder of the form `plot_*.py` will be executed and their generated
plots will be available for viewing in the `/auto_examples` URL.

To build the documentation locally execute

```shell
cd doc
make html
```

The project uses [CircleCI](https://circleci.com/) to build its documentation
from the `master` branch and host it using [GitHub Pages](https://pages.github.com/).
