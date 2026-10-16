.PHONY: install lint format test serve train docker

install:
	pip install -e ".[train,api,dev]"

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

test:
	pytest

serve:
	enfr serve

train:
	enfr train --data $(DATA) --epochs 1

docker:
	docker compose up --build
