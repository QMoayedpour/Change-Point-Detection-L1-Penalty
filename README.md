# Change-Point Detection (L0 Penalty)

Small research project reproducing the CPOP algorithm from *Detecting Changes in Slope With an L0 Penalty* (Maidstone et al., 2017). It also includes a PELT baseline, synthetic data generation, and the write-up used for a class project.

## What’s here
- Core implementation: `src/cpop.py` (CPOP) and `src/pelt.py` (PELT baseline).
- Cost helpers and synthetic data utilities in `src/cost.py` and `src/utils.py`.
- Example datasets in `data/` (GDP and market-price series).
- Exploration notebooks: `init.ipynb` and `ntbk/main.ipynb`.
- Project report and slides: `report.pdf`, `slides_presentation.pdf`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows
pip install -e .
```

## Quick start
```python
import numpy as np
from src.cpop import CPOP
from src.utils import generate_processus

# Synthetic piecewise-linear series with Gaussian noise
y = generate_processus(T=200, k=4, sigma=3, scale=2)

# Run CPOP (beta tunes how aggressively changepoints are penalized)
model = CPOP(y, beta=20)
changepoints = model.run()

# Optional: build the fitted approximation and plot it
approx = model.compute_approx_and_plot(verbose=False, logs=True)
print("Changepoints (inclusive indices):", changepoints)
```

### Using your own data
Pass any 1D NumPy array to `CPOP(y, beta=..., sigma=...)`. If `sigma` is not provided, it is estimated from the median absolute deviation of the first differences. For visual inspection, call `compute_approx_and_plot(verbose=True)` to overlay the fitted signal and changepoints.

## Reference
- Maidstone, R., Fearnhead, P., Letchford, A., & Hocking, T. (2017). *Detecting Changes in Slope With an L0 Penalty*. [arXiv:1701.01672](https://arxiv.org/pdf/1701.01672)
