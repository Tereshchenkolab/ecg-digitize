# Run unit tests
test:
	python -m pytest ./tests

# Install dependencies needed to run
build:
	pip install -r requirements.txt

# Install extra packages required for development
develop:
	pip install -r requirements-development.txt

typecheck:
	mypy digitize/
