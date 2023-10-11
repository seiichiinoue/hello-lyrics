# hello-lyrics

## Quick start

Install `poetry`: [docs](https://python-poetry.org/docs/#installation)

Install dependencies:

```sh
poetry install
```

Convert data:

```sh
poetry run python convert_data.py
```

Visualize:

```sh
poetry run tensorboard --logdir log_dir 
```