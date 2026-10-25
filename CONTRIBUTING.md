# Contributing

Use Python 3.11 or newer. Install the locked development environment with `uv sync --locked --extra dev`.
Run `uv run ruff check src tests examples`, `uv run ruff format --check src tests examples` and `uv run pytest -q`.

A physical change needs a meaningful reference: a conservation identity, closed form, independent numerical formulation, identified data, or measured fixture. A test that merely repeats implementation arithmetic is insufficient.

Declare units, signs, numerical method, assumptions, and supported parameter ranges. Retain infeasible cases as limits rather than replacing them with successful-looking zeros. Separate model verification from real-device validation.

Make each commit a coherent model, check, interface, study, or documentation change. Do not use empty commits or manufactured dates. Rebuild galleries from one clean source revision after changing physical studies.
