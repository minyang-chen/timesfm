pyenv install 3.11
pyenv versions # to list the versions available (lets assume the versions are 3.10.15 and 3.11.10)

# For PyTorch version installation do the following.

pyenv local 3.11.12
poetry env use 3.11.12
poetry lock
poetry install -E  torch
pip install poetry

# covariable support
pip install jax jaxlib

# Install TimesFM with PyTorch support
pip install timesfm[torch]


## Usage
```
# For Torch
import timesfm
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
  )

import numpy as np
forecast_input = [
    np.sin(np.linspace(0, 20, 100)),
    np.sin(np.linspace(0, 20, 200)),
    np.sin(np.linspace(0, 20, 400)),
]
frequency_input = [0, 1, 2]

point_forecast, experimental_quantile_forecast = tfm.forecast(
    forecast_input,
    freq=frequency_input,
)

```