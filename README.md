# MPPI4Rob (minimal)

Minimal MPPI controller demo (PyTorch).

Dependencies
------------
Install the minimal dependencies (CPU or GPU as available):

```bash
pip install torch numpy matplotlib gymnasium
```

Run example
-----------
Run the compact pendulum example (installed deps required):

```bash
python3 examples/pendulum_example.py
```

Run tests
---------
This repository includes a small pytest to verify the controller runs:

```bash
pip install pytest
pytest -q
```

Test file: `tests/test_mppi_basic.py`

Notes
-----
- The canonical controller implementation is `controllers/mppi_controller.py`.
- `controllers/mppi_controller_improved.py` is kept as a tiny stub to avoid duplication.
- If you want the absolute minimal set (no packaging files), tell me and I will remove them.
