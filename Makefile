.PHONY: wheel develop test format lint typecheck

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
	clang-format -i src/*.cpp
	prek run --all-files ruff-format

# prek の ruff フックは git 追跡下のファイルのみを対象にするため、 未追跡ファイルは検査されない。
lint:
	prek run --all-files ruff-check

# ty はプロジェクト全体を走査する。 生成スタブ (src/libdatachannel/__init__.pyi) が無いと
# libdatachannel を解決できないため、 事前に make develop を実行しておく。
typecheck:
	prek run --all-files ty
