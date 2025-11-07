# AGENTS
## Build / Run
- `pixi run python gradient_boosted_trees_flow.py`
- `pixi run python neural_network_flow.py`
- Or `metaflow run <FlowName>` with parameters.
## Lint
- `pixi run pylint *.py` (add a `.pylintrc` for custom rules).
## Test
- Full suite: `pixi run pytest`.
- Single test: `pixi run pytest path/to/test_file.py::test_name`.
## Code style
- Imports: stdlib → third‑party → local, one per line, absolute.
- Formatting: 4 spaces, max line length 88, trailing commas on multi‑line literals.
- Types: annotate public functions (`def f(a: int) -> str:`).
- Naming: modules/files `snake_case`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE_CASE`.
- Errors: raise specific exceptions, use `try/except` with context; avoid bare `except:`.
## Cursor / Copilot
- No `.cursor`/`.cursorrules`; if added, follow them.
- If `.github/copilot-instructions.md` exists, obey its guidelines.