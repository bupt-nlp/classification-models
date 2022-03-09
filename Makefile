SOURCE_GLOB=$(wildcard src/**/*.py src/*.py)

IGNORE_PEP=E203,E221,E241,E272,E501,F811,C0301,C0114

export PYTHONPATH=./

.PHONY: pylint
pylint:
	pylint \
		--load-plugins pylint_quotes \
		--disable=W0511,R0801,cyclic-import,C0301,C0303,C0114,C4001,W0221,C0115,R0903,C0116,E1121 \
		$(SOURCE_GLOB)


.PHONY: lint
lint: pylint

.PHONY: install
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

.PHONY: pytest
pytest:
	pytest src/ tests/

.PHONY: test
test: lint pytest