create_env:
	conda create -y -n vhsven-sklearn python=3.6 cython graphviz jupyter matplotlib nose numpydoc pandas pep8 pillow pylint pytest rope scikit-learn seaborn sphinx
	source activate vhsven-sklearn
	pip install doc8 restructuredtext-lint sphinx-gallery sphinx-rtd-theme
	conda develop /path/to/vhsven-sklearn

activate_env:
	source activate vhsven-sklearn

export_env:
	conda env export > environment_dev.yml
