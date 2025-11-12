# MPPI4Rob — Minimal MPPI controller demo

Lightweight, easy-to-read MPPI (Model Predictive Path Integral) controller implemented with PyTorch.
This repository aims to be a compact starting point for experimenting with MPPI and small robotics examples.

Key features
------------
- Numerical stability improvements (log-space weight computation).
- Exploration control: adaptive temperature (entropy-based) and noise annealing (cosine decay).
- Optional residual-MPPI mode: MPPI can optimize a residual action on top of a user-provided baseline policy.
- Minimal, well-separated interfaces for dynamics, cost, baseline policy and observation functions.

Dependencies
------------
Install the runtime dependencies (CPU or GPU builds of PyTorch as needed):

```bash
pip install torch numpy matplotlib gymnasium
```

Examples
--------
Two compact examples are included under `examples/`:

- `examples/pendulum_example.py`
	- Basic MPPI controlling the OpenAI Gym `Pendulum-v1` environment.
	- Usage:
		```bash
		python3 examples/pendulum_example.py
		```

- `examples/pendulum_residual_example.py`
	- Demonstrates residual MPPI: a simple PD baseline policy is wrapped as a `BaselinePolicy` and MPPI optimizes a residual action around it.
	- The script accepts a `--mode` argument to choose what action the controller returns:
		- `--mode combined` (or omit `--mode`): baseline + residual (full action) — default when residuals are enabled.
		- `--mode residual`: MPPI returns residual-only (you can add baseline yourself before sending to the env).
		- `--mode baseline`: run the baseline controller only (no MPPI sampling).
	- Usage examples:
		```bash
		# baseline + residual (default)
		python3 examples/pendulum_residual_example.py --mode combined

		# baseline-only
		python3 examples/pendulum_residual_example.py --mode baseline

		# residual-only
		python3 examples/pendulum_residual_example.py --mode residual
		```

Usage notes
-----------
- The canonical controller implementation is `controllers/mppi_controller.py`.
- To run the examples you must have the dependencies installed (see above). If running on a headless server you can disable rendering with the `--render` flag omitted (or in `pendulum_example.py` pass `--headless` if added).

License & contribution
----------------------
This project is intentionally small and permissive. Feel free to open issues or PRs to add small examples or improvements.
