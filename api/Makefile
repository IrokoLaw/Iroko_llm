# Makefile
.ONESHELL:

.PHONY: api
build:
	docker build -t iroko-llm .

run:
	docker run -d \
		--name iroko-llm \
		-v ${DATA_FILE}:/app/data \
		-p 8000:8000 \
		iroko-llm