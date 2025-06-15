pyenv install 3.10
pyenv versions # to list the versions available (lets assume the versions are 3.10.15 and 3.11.10)

#For PAX version installation do the following.

pyenv local 3.10.17
poetry env use 3.10.17
poetry lock
poetry install -E  paxpip install poetry

pip install poetry

## Usage
```
import timesfm

# Loading the timesfm-2.0 checkpoint:
# For PAX
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
          num_layers=50,
          context_len=2048,

          use_positional_embedding=False,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-jax"),
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
