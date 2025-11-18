# AGENTS
## Build / Run
- `pixi run python gradient_boosted_trees_flow.py run`
- `pixi run python neural_network_zen.py`
- Start the zenml server with `pixi run zenml login --local --ip-address 0.0.0.0` but only if it is not already running. When you are done, run `pixi run zenml logout` to tear it down.
- Start the metaflow-dev server with `pixi run metaflow-dev up`, then run `pixi run metaflow-dev shell` in the terminal to log results. When you are done run `pixi run metaflow-dev down` to tear it down.
## Package Management
- If you need to check what packages are available, read @pixi.toml.
- If you need to add a package, run `pixi add --pypi <package>`. Multiple packages can be listed with spaces between them in a single command.
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
- Logging and Print Statements: Use this color scheme for all logs:
  - Green: Success
  - Yellow: Warnings
  - Red: Failure, Errors
  - Blue: Diagnostics, Information
  - Purple: All other text
  - White: ASCII Art
## Cursor / Copilot
- No `.cursor`/`.cursorrules`; if added, follow them.
- If `.github/copilot-instructions.md` exists, obey its guidelines.
