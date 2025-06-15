import timesfm

# Loading the timesfm-2.0 checkpoint:
# For Torch
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

# Examples:
# Array inputs, with the frequencies set to low, medium and high respectively.
# frequency: 
#     0: T, MIN, H, D, B, U
#     1: W, M
#     2: Q, Y

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
print("Point Forecast:", point_forecast)

# import pandas as pd
# forecast_df = tfm.forecast_on_df(
#     inputs=input_df,
#     freq="M",  # monthly
#     value_name="y",
#     num_jobs=-1,
# )
