#!/usr/bin/env python3
"""CLI entry point for the tutorial coverage validation suite.

What is Being Computed?:
    Triggers the validation of the tutorial coverage JSON metadata against the 
    actual Jupyter notebooks in the repository.

Algorithm:
    1. Import the `main` entry point from `pysurgery.utils.tutorial_coverage_validator`.
    2. Execute the validation suite.
    3. Propagate the exit code (0 for success, non-zero for failure).

Use When:
    - Verifying documentation integrity from the command line.
    - Integrating into pre-commit hooks or CI workflows.
"""
from pysurgery.utils.tutorial_coverage_validator import main


if __name__ == "__main__":
    raise SystemExit(main())
