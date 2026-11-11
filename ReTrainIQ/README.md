# ReTrainIQ

Continuous Model Monitoring, CARA-T Staleness Detection, Drift Analysis & Adaptive Retraining Pipeline.

Python 3.10 or newer (tested on 3.10).

## Setup Instructions

1. Create and activate a Python virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup environment and Python path:
```bash
python 0-Init/ri_init/setup_env.py
```

4. Run tests:
```bash
pytest -q
```
