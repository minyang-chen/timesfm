import timesfm
timesfm_backend = "gpu"  # @param

model = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend=timesfm_backend,
          per_core_batch_size=32,
          horizon_len=128,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
  )

## checkpoint ajax=google/timesfm-2.0-500m-jax
## checkpoint pytorch=google/timesfm-2.0-500m-pytorch

# # Covariates

# Let's take a toy example of forecasting sales for a grocery store: 

# **Task:** Given the observed the daily sales of this week (7 days), forecast the daily sales of next week (7 days).

# ```
# Product: ice cream
# Daily_sales: [30, 30, 4, 5, 7, 8, 10]
# Category: food
# Base_price: 1.99
# Weekday: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
# Has_promotion: [Yes, Yes, No, No, No, Yes, Yes, No, No, No, No, No, No, No]
# Daily_temperature: [31.0, 24.3, 19.4, 26.2, 24.6, 30.0, 31.1, 32.4, 30.9, 26.0, 25.0, 27.8, 29.5, 31.2]
# ```

# ```
# Product: sunscreen
# Daily_sales: [5, 7, 12, 13, 5, 6, 10]
# Category: skin product
# Base_price: 29.99
# Weekday: [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
# Has_promotion: [No, No, Yes, Yes, No, No, No, Yes, Yes, Yes, Yes, Yes, Yes, Yes]
# Daily_temperature: [31.0, 24.3, 19.4, 26.2, 24.6, 30.0, 31.1, 32.4, 30.9, 26.0, 25.0, 27.8, 29.5, 31.2]
# ```

# In this example, besides the `Daily_sales`, we also have covariates `Category`, `Base_price`, `Weekday`, `Has_promotion`, `Daily_temperature`. Let's introduce some concepts:

# **Static covariates** are covariates for each time series. 
# - In our example, `Category` is a **static categorical covariate**, 
# - `Base_price` is a **static numerical covariates**.

# **Dynamic covariates** are covaraites for each time stamps.
# - Date / time related features can be usually treated as dynamic covariates.
# - In our example, `Weekday` and `Has_promotion` are **dynamic categorical covariates**.
# - `Daily_temperate` is a **dynamic numerical covariate**.

# **Notice:** Here we make it mandatory that the dynamic covariates need to cover both the forecasting context and horizon. For example, all dynamic covariates in the example have 14 values: the first 7 correspond to the observed 7 days, and the last 7 correspond to the next 7 days.

# TimesFM with Covariates

## The strategy we take here is to treat covariates as batched in-context exogenous regressors (XReg) and fit linear models on them outside of TimesFM. The final forecast will be the sum of the TimesFM forecast and the linear model forecast.
##  In simple words, we consider these two options.
## **Option 1:** Get the TimesFM forecast, and fit the linear model regressing the residuals on the covariates ("timesfm + xreg").
## **Option 2:** Fit the linear model of the time series itself on the covariates, then forecast the residuals using TimesFM  ("xreg + timesfm").
## Let's take a code at the example of Electricity Price Forecasting (EPF). 

import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv('https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE.csv')
df['ds'] = pd.to_datetime(df['ds'])
print(df.head())

# This dataset has a few covariates beside the hourly target `y`:
# - `unique_id`: a static categorical covariate indicating the country.
# - `gen_forecast`: a dynamic numerical covariate indicating the estimated electricity to be generated.
# - `system_load`: the observed system load. Notice that this **CANNOT** be considered as a dynamic numerical covariate because we cannot know its values over the forecasting horizon in advance.
# - `weekday`: a dynamic categorical covariate.
# Let's now make some forecasting tasks for TimesFM based on this dataset. For simplicity we create forecast contexts of 120 time points (hours) and forecast horizons of 24 time points.

# Data pipelining
def get_batched_data_fn(
    batch_size: int = 128, 
    context_len: int = 120, 
    horizon_len: int = 24,
):
  examples = defaultdict(list)

  num_examples = 0
  for country in ("FR", "BE"):
    sub_df = df[df["unique_id"] == country]
    for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
      num_examples += 1
      examples["country"].append(country)
      examples["inputs"].append(sub_df["y"][start:(context_end := start + context_len)].tolist())
      examples["gen_forecast"].append(sub_df["gen_forecast"][start:context_end + horizon_len].tolist())
      examples["week_day"].append(sub_df["week_day"][start:context_end + horizon_len].tolist())
      examples["outputs"].append(sub_df["y"][context_end:(context_end + horizon_len)].tolist())
  
  def data_fn():
    for i in range(1 + (num_examples - 1) // batch_size):
      yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}
  
  return data_fn


# Define metrics
def mse(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

# Now let's try `model.forecast_with_covariates`. 
# In particular, the output is a tuple whose first element is the new forecast.

import time

# Benchmark
batch_size = 128
context_len = 120
horizon_len = 24
input_data = get_batched_data_fn(batch_size = 128)
metrics = defaultdict(list)


for i, example in enumerate(input_data()):
  raw_forecast, _ = model.forecast(
      inputs=example["inputs"], freq=[0] * len(example["inputs"])
  )
  start_time = time.time()
  # Forecast with covariates
  # Output: new forecast, forecast by the xreg
  cov_forecast, ols_forecast = model.forecast_with_covariates(  
      inputs=example["inputs"],
      dynamic_numerical_covariates={
          "gen_forecast": example["gen_forecast"],
      },
      dynamic_categorical_covariates={
          "week_day": example["week_day"],
      },
      static_numerical_covariates={},
      static_categorical_covariates={
          "country": example["country"]
      },
      freq=[0] * len(example["inputs"]),
      xreg_mode="xreg + timesfm",              # default
      ridge=0.0,
      force_on_cpu=False,
      normalize_xreg_target_per_input=True,    # default
  )
  print(
      f"\rFinished batch {i} linear in {time.time() - start_time} seconds",
      end="",
  )
  metrics["eval_mae_timesfm"].extend(
      mae(raw_forecast[:, :horizon_len], example["outputs"])
  )
  metrics["eval_mae_xreg_timesfm"].extend(mae(cov_forecast, example["outputs"]))
  metrics["eval_mae_xreg"].extend(mae(ols_forecast, example["outputs"]))
  metrics["eval_mse_timesfm"].extend(
      mse(raw_forecast[:, :horizon_len], example["outputs"])
  )
  metrics["eval_mse_xreg_timesfm"].extend(mse(cov_forecast, example["outputs"]))
  metrics["eval_mse_xreg"].extend(mse(ols_forecast, example["outputs"]))

print()

for k, v in metrics.items():
  print(f"{k}: {np.mean(v)}")

# You should see results close to 
# ```
# eval_mae_timesfm: 6.729583250571446
# eval_mae_xreg_timesfm: 5.3375301110158
# eval_mae_xreg: 37.152760709266
# eval_mse_timesfm: 162.3132151851567
# eval_mse_xreg_timesfm: 120.9900627409689
# eval_mse_xreg: 1672.208769045399
# ```
# With the covariates, the TimesFM forecast Mean Absolute Error improves from 6.73 to 5.34, and Mean Squred Error from 162.31 to 120.99. The results of purely fitting the linear model are also provided for reference.


## Formatting Your Request
#It is quite crucial to get the covariates properly formatted so that we can call this `model.forecast_with_covariates`. Please see its docstring for details. Here let's also grab a batch from a toy data input pipeline for quick explanations.

toy_input_pipeline = get_batched_data_fn(batch_size=2, context_len=5, horizon_len=2)
print(next(toy_input_pipeline()))

# You should see something similar to this
# ```
# {
#     'country': ['FR', 'FR'], 
#     'inputs': [[53.48, 51.93, 48.76, 42.27, 38.41], [48.76, 42.27, 38.41, 35.72, 32.66]], 
#     'gen_forecast': [[76905.0, 75492.0, 74394.0, 72639.0, 69347.0, 67960.0, 67564.0], [74394.0, 72639.0, 69347.0, 67960.0, 67564.0, 67277.0, 67019.0]], 
#     'week_day': [[3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3]], 
#     'outputs': [[35.72, 32.66], [32.83, 30.06]],
# }
# ```

# Notice:
# - We have two examples in this batch.
# - For each example we support different context lengths and horizon lengths just as `model.forecast`. Although it is not demonstrated in this dataset.
# - If dynamic covariates are present, the horizon lengths will be inferred from them, e.g. how many values are provided in additional to the ones corresponding to the inputs. Make sure all your dynamic covariates have the same length per example.
# - The static covariates are one per example.

