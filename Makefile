.PHONY: wheel develop test format

wheel:
	uv build --wheel

develop: wheel
	uv pip install -e . --force-reinstall
	@echo "Copying .pyi stub file..."
	@cp _build/__init__.pyi src/libdatachannel/ 2>/dev/null || true

test: develop
	uv run pytest tests/

example:
	uv sync --group example

format:
	clang-format -i src/*.cpp src/*.h
	uv run ruff format tests/ examples/