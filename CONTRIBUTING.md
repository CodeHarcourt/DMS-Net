# Contributing to DMS-Net

Thanks for taking the time to contribute! Here are a few guidelines to keep in
mind while working on this project.

## Getting started

1. Fork the repository and clone your fork locally.
2. Create a virtual environment and install the dependencies from
   `requirements.txt`.
3. Create a feature branch from `main` (e.g. `feature/add-cosine-schedule`).
4. Make your changes and ensure the code still runs.
5. Open a Pull Request (PR) against the upstream repository.

## Coding standards

- Keep modules, functions and variables clearly named in English.
- Use docstrings for public classes and functions.
- Keep code comments in English.
- Prefer simple, readable solutions over clever ones.

## Pull request checklist

Before submitting your PR, please verify the following:

- [ ] The code runs locally with the supported Python version.
- [ ] Any new modules are exported where appropriate (`__init__.py`, docs).
- [ ] You have updated or added comments, docstrings and README sections as
      needed.
- [ ] You have not committed model checkpoints, dataset files or other large
      binary artifacts.

## Opening issues

When filing a bug report, please include:

- Environment details (Python, PyTorch, CUDA versions)
- A minimal reproducible command / snippet
- Expected vs. actual behavior

Thanks again for your contribution!
