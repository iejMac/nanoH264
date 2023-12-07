install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

test: ## [Local development] Run unit tests
	pytest -s tests/
