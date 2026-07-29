# Testing Guide

## Overview
Tests in stable-pretraining are categorized to separate fast unit tests from slow integration tests. This enables efficient CI/CD while maintaining comprehensive test coverage.

## Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Fast, no GPU, no downloads
- **Integration Tests** (`@pytest.mark.integration`): Full training runs
- **Regression Tests** (`@pytest.mark.regression`): One seeded epoch per method on fake data
- **GPU Tests** (`@pytest.mark.gpu`): Require CUDA
- **Slow Tests** (`@pytest.mark.slow`): Take > 1 minute
- **Download Tests** (`@pytest.mark.download`): Download data

## Local Environment

The integration and regression jobs both install `requirements-ci.txt` before
the package, so a local checkout needs the same versions to reproduce what CI
sees. `.venv/` is gitignored; every dev box keeps one at the repo root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-ci.txt
uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate
```

## Running Tests

```bash
# Unit tests only (CI default)
python -m pytest -m unit

# Integration tests
python -m pytest -m integration

# Regression tests
python -m pytest stable_pretraining/tests/ -m regression

# All tests
python -m pytest

# Exclude slow tests
python -m pytest -m "not slow"
```

## Test Organization

```
stable_pretraining/tests/
├── conftest.py              # Pytest configuration
├── utils.py                 # Test utilities and mocks
├── test_*_unit.py          # Unit tests
├── test_*_integration.py   # Integration tests
└── test_*.py               # Original tests
```

## Writing Unit Tests

### Use Test Utilities
```python
from stable_pretraining.tests.utils import MockImageDataset, create_mock_dataloader

@pytest.mark.unit
def test_loss_computation():
    # Test just the loss computation
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss_fn = spt.losses.NTXEntLoss(temperature=0.1)
    loss = loss_fn(z1, z2)
    assert loss.item() >= 0
```

### Best Practices

1. **Mock Dependencies**: Use `MockImageDataset` instead of real downloads
2. **Small Tensors**: Use `batch_size=4, image_size=32` for speed
3. **Fast Execution**: Tests should run in < 1 second
4. **Single Responsibility**: Test one component at a time
5. **Proper Markers**: Always mark tests appropriately

### Example Refactoring

**Before** (Integration test):
```python
def test_simclr():
    # Downloads ImageNette, requires GPU, full training
    train_dataset = spt.data.HFDataset(
        path="frgfm/imagenette",
        name="160px",
        split="train",
        transform=train_transform,
    )
    # ... full training loop
```

**After** (Unit test):
```python
@pytest.mark.unit
def test_simclr_loss():
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss_fn = spt.losses.NTXEntLoss(temperature=0.1)
    loss = loss_fn(z1, z2)
    assert loss.item() >= 0

@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
def test_simclr_training():
    # Original full test with proper markers
```

## CI Configuration

GitHub Actions runs only unit tests:
```yaml
- name: Run Unit Tests
  run: python -m pytest stable_pretraining/ -m unit --verbose --cov=stable_pretraining --cov-report term
```
