# [<img alt="Frontx logo" src="https://raw.githubusercontent.com/gerlero/frontx/main/logo.png" width=300></img>](https://github.com/gerlero/frontx)

**[JAX](https://github.com/jax-ml/jax)-accelerated numerical library for nonlinear diffusion problems in semi-infinite domains**

[![Documentation](https://img.shields.io/readthedocs/frontx)](https://frontx.readthedocs.io/)
[![CI](https://github.com/gerlero/frontx/actions/workflows/ci.yml/badge.svg)](https://github.com/gerlero/frontx/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/gerlero/frontx/branch/main/graph/badge.svg)](https://codecov.io/gh/gerlero/frontx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Publish](https://github.com/gerlero/frontx/actions/workflows/pypi-publish.yml/badge.svg)](https://github.com/gerlero/frontx/actions/workflows/pypi-publish.yml)
[![PyPI](https://img.shields.io/pypi/v/frontx)](https://pypi.org/project/frontx/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/frontx)](https://pypi.org/project/frontx/)

## 📦 Installation

Install with pip:

```bash
pip install frontx
```

## ⚙️ Enabling 64-bit (double precision) support in JAX

Some problems require double precision (i.e., `float64`) types to be solvable. By default, JAX uses 32-bit single precision numbers for performance reasons.

To enable 64-bit support, use the following code snippet:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

☝️ **Important**:
- This setting must be applied at the very beginning of your script.
- JAX does not persist this configuration between runs: you'll need to do this each time you start a new session.