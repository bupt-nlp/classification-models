
SOURCE_GLOB=$(wildcard src/**/*.py src/*.py)

IGNORE_PEP=E203,E221,E241,E272,E501,F811,C0301,C0114

.PHONY: pylint
pylint:
	pylint \
		--load-plugins pylint_quotes \
		--disable=W0511,R0801,cyclic-import,C0301,C0114,C4001 \
		$(SOURCE_GLOB)

.PHONY: pycodestyle
pycodestyle:
	pycodestyle \
		--statistics \
		--count \
		--ignore="${IGNORE_PEP}" \
		$(SOURCE_GLOB)

.PHONY: lint
lint: pylint pycodestyle

.PHONY: install
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt