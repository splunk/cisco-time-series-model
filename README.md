# Cisco Time Series Model 1.0

[![arXiv technical report](https://img.shields.io/static/v1?label=Technical-Report&message=2511.19841&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2511.19841)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-FFD21E)](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

The Cisco Time Series Model is a foundation model trained to perform univariate zero-shot forecasting. Its core is a sequence of decoder-only transformer layers. It is architecturally inspired by the [TimesFM2.0 model](https://huggingface.co/google/timesfm-2.0-500m-pytorch), with multiresolution modifications aimed at efficient use of long context. It expects a multiresolution context (x<sub>c</sub>, x<sub>f</sub>), where the resolution (i.e., space between data points) of x<sub>c</sub> is 60 times the resolution of x<sub>f</sub>. Both x<sub>c</sub> and x<sub>f</sub> can have length up to 512. The input contexts should be aligned “on the right,” e.g., if x<sub>f</sub> consists of the 512 minutes terminating at 11:00AM on November 11, then x<sub>c</sub> should consist of the 512 hours terminating at the same time. The output is a forecast of 128 points, which should be interpreted at the finer resolution; and corresponding quantiles for these points.

For convenience, we also provide utilities for preparing a multiresolution context from a single resolution context (with length up to 512 x 60 = 30,720) directly.

## Latest Release (250M) - [CTSM 1.0](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0)
Compared to our November 2025 ([`cisco-time-series-model-1.0-preview`](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0-preview)) release, the 250M model:
- Is trained **from scratch** (no continued pretraining (CPT) from TimesFM weights).
- Uses **2× internal observability (O11y) data** in the training mixture.
- Improves **GIFT-Eval (public) MASE by 8.6%**.
- Reduces size from **~500M → ~250M parameters**.
- Adds **short-context training** and **bidirectional coarse attention** for better robustness on short histories.


## Model Architecture and Training Details
<figure>
  <img src="https://raw.githubusercontent.com/splunk/cisco-time-series-model/main/1.0/images/ctsm-rc-1_0.png" alt="Multiresolution model architecture">
  <figcaption><em>Architecture diagram illustrating our novel additions.</em></figcaption>
</figure>

Our latest [Cisco Time Series Model 1.0](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0) is a 250M parameter model that is trained from scratch and uses 25 transformer layers (vs 50 in [`1.0-preview`](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0-preview)). This checkpoint leverages the architectural changes of expanded quantiles (15 quantile outputs, 0.01-0.99), rotary positional embeddings (RoPE), special token, separate resolution embeddings for each of coarse context / fine context / special token, and bidirectional attention over the coarse-resolution context. 

### Standard Causal Attention vs Bidirectional Coarse Attention (Ours):
<figure>
  <img src="https://raw.githubusercontent.com/splunk/cisco-time-series-model/main/1.0/images/bidirectional_coarse_attention.png" alt="Bidirectional coarse attention illustration">
  <figcaption><em>A representation of bidirectional attention over the coarse context, where the model attends to both left and right context within the coarse context, while still attending causally to the fine context.</em></figcaption>
</figure>

### Example Visualization of Multiresolution Time Series Input to the Model
<figure>
  <img src="https://raw.githubusercontent.com/splunk/cisco-time-series-model/main/1.0/images/multi_resolution_time_series_example.png" alt="Multiresolution time series example with padded 1-hour context">
  <figcaption><em>Multiresolution time series example with padded 1-hour context.</em></figcaption>
</figure>

### Training Datasets
The distribution of training corpus is as follows:
- Internal Splunk Observability Cloud metric time series (upweighted ~2× vs the previous release) datasets:
  - (1-hour, 1-minute) resolution: **46.2%**
  - (5-hour, 5-minute) resolution: **21.8%**
- Public datasets:
  - [GIFT-EVAL](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain) : **19.5%**
  - [Chronos](https://huggingface.co/datasets/autogluon/chronos_datasets) : **3%**
- Synthetic dataset: **9.5%**

## Technical Report
- A detailed technical report is available on [arXiv](https://arxiv.org/abs/2511.19841) (also available locally [here](https://github.com/splunk/cisco-time-series-model/blob/main/1.0-preview/technical_report/Cisco-Time-Series-Model-Technical-Report.pdf))<br>

**NOTE:** the report focuses on our November 2025 release (`1.0-preview`), so some training details differ from our latest release. An updated technical report with details on the latest release will be made available in the near future.

## [GIFT-EVAL](https://github.com/SalesforceAIResearch/gift-eval) Benchmark
All results below are computed on the entire [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) benchmark, with forecast horizon capped at ≤ 128. Models are sorted by MASE scores (lower is better).

#### 1. Using the entire time series context
| MODEL | MSE | MAE | MASE | sMAPE | MSIS | CRPS | Test Leak. |
|:---|:---|:---|:---|:---|:---|:---|:---|
| [`Chronos-2`](https://huggingface.co/amazon/chronos-2) | `0.471` | `0.659` | `0.678` | `0.887` | `0.464` | `0.531` | `NO` |
| [`TimesFM-2.5`](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | `0.464` | `0.667` | `0.684` | `0.851` | `0.494` | `0.540` | `NO` |
| [`TimesFM-2.0`](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | `0.521` | `0.697` | `0.707` | `0.933` | `0.529` | `0.566` | `YES` |
| [`Toto_Open_Base_1.0`](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | `0.528` | `0.691` | `0.715` | `0.932` | `0.526` | `0.559` | `NO` |
| [`CTSM-1.0 (Ours)`](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0) | `0.510` | `0.692` | `0.715` | `0.936` | `0.538` | `0.564` | `NO` |
| [`Chronos-bolt-base`](https://huggingface.co/amazon/chronos-bolt-base) | `0.562` | `0.722` | `0.747` | `0.967` | `0.597` | `0.592` | `YES` |
| [`CTSM-1.0-preview (Ours, prev.)`](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0-preview) | `0.585` | `0.757` | `0.782` | `0.999` | `0.584` | `0.615` | `YES` |

#### 2. Using the time series context truncated to 1024 points
| MODEL | MSE | MAE | MASE | sMAPE | MSIS | CRPS | Test Leak. |
|:---|:---|:---|:---|:---|:---|:---|:---|
| [`Chronos-2`](https://huggingface.co/amazon/chronos-2) | `0.493` | `0.674` | `0.692` | `0.904` | `0.485` | `0.542` | `NO` |
| [`TimesFM-2.5`](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | `0.493` | `0.684` | `0.707` | `0.857` | `0.530` | `0.556` | `NO` |
| [`CTSM-1.0 (Ours)`](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0) | `0.510` | `0.692` | `0.715` | `0.936` | `0.538` | `0.564` | `NO` |
| [`TimesFM-2.0`](https://huggingface.co/google/timesfm-2.0-500m-pytorch) | `0.532` | `0.704` | `0.730` | `0.950` | `0.563` | `0.573` | `YES` |
| [`Toto_Open_Base_1.0`](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | `0.573` | `0.715` | `0.738` | `0.954` | `0.549` | `0.580` | `NO` |
| [`Chronos-bolt-base`](https://huggingface.co/amazon/chronos-bolt-base) | `0.552` | `0.721` | `0.749` | `0.967` | `0.591` | `0.589` | `YES` |
| [`CTSM-1.0-preview (Ours, prev.)`](https://huggingface.co/cisco-ai/cisco-time-series-model-1.0-preview) | `0.585` | `0.757` | `0.782` | `0.999` | `0.584` | `0.615` | `YES` |

> ***Test Leak.:** `YES` indicates that the model's training data inadvertently includes portions of the [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) test set.*

## Usage notes
- If the input time series is missing some values, imputation via last value is recommended; if the time series is naturally sparse and this leads to excessive imputation (e.g., more than 30% of values are imputed), the model forecasts will deteriorate.
- This release includes short-context training with 1/3rd of training data sampled uniformly in the range of `[10, 511]` in the fine context. However, the model generally works better when more coarse resolution history is provided.
- The quantiles have not been calibrated or rigorously evaluated, e.g., we currently do not have evidence to support a claim along the lines of “the range from q=0.01 to q=0.99 contains the true value 98% of the time (under some mild conditions).”

## Dependencies and Installation

### Setup Virtual Environment
Create a Python 3.11 (recommended) virtual environment and install the package using `uv`:
```shell
# Install uv if required
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate
```

### PyPI Package
The Cisco Time Series Model is available as a PyPI package named `cisco-tsm`. You can install it using pip:
```shell
uv pip install cisco-tsm
```

### Manual Installation
1.  Clone the repository:
    ```shell
    git clone https://github.com/splunk/cisco-time-series-model.git
    cd cisco-time-series-model
    ```

2.  [Optional] Install your preferred `torch` backend based on your OS and accelerators (CPU, GPU, TPU or Apple Silicon):
    - [Install PyTorch](https://pytorch.org/get-started/locally/).

3. Install the package:
    ```shell
    uv pip install -e .
    ```

4.  _(Deprecated)_ If you want to use the `1.0-preview` model, change directory to `1.0-preview/` and use the corresponding package (follow its README for installation and usage):
    ```shell
    cd 1.0-preview/
    ```

## Example Code

```python
import torch
import numpy as np
from cisco_tsm import CiscoTsmMR, TimesFmHparams, TimesFmCheckpoint

rng = np.random.default_rng(42)

## Sample data
T = 512 * 60
hours = (T + 59) // 60
k = np.arange(hours, dtype=np.float32)
h = (80 + 0.1 * k) * (1 + 0.25 * np.sin(2 * np.pi * k / 24))
t = np.arange(T, dtype=np.float32)

input_series_1 = h[(t // 60).astype(int)] * (1 + 0.05 * np.sin(2 * np.pi * t / 30)) + rng.normal(0, 0.4, size=T)

# Hyperparameters
hparams = TimesFmHparams(
    num_layers=25, # For `1.0-preview`, use `num_layers=50`.
    use_positional_embedding=False,
    backend="gpu" if torch.cuda.is_available() else "cpu",
    quantiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
)

ckpt = TimesFmCheckpoint(huggingface_repo_id="cisco-ai/cisco-time-series-model-1.0") # For `1.0-preview`, use `huggingface_repo_id="cisco-ai/cisco-time-series-model-1.0-preview"`.

model = CiscoTsmMR(hparams=hparams, checkpoint=ckpt)

# Model Inference
forecast_preds = model.forecast(input_series_1, horizon_len=128)

# Access forecast mean and quantiles of each series
mean_forecast = forecast_preds[0]['mean'] # (128,)
quantiles = forecast_preds[0]['quantiles'] # dict with keys as quantile levels (0.01, 0.05, ...., 0.95, 0.99) and values as (128,) numpy arrays

# You can also forecast multiple series at once
T = 25_000
hours = (T + 59) // 60
k = np.arange(hours, dtype=np.float32)
h = 120 / (1 + np.exp(-0.01 * (k - 300))) + 10 * np.cos(2 * np.pi * k / (24*7))
t = np.arange(T, dtype=np.float32)
input_series_2 = h[(t // 60).astype(int)] + 2 * np.sin(2 * np.pi * t / 60) + rng.normal(0, 0.5, size=T)

multi_series_forecasts = model.forecast([input_series_1, input_series_2], horizon_len=128)

# Long horizon forecasting is also supported and can be invoked as follows
long_horizon_forecasts = model.forecast(input_series_1, horizon_len=240)

```

## Example Notebooks
We also provide a few Jupyter notebooks demonstrating how to use the Cisco Time Series Model for forecasting real-world time series data:
- [CPU Utilization Forecasting](https://github.com/splunk/cisco-time-series-model/blob/main/1.0/notebooks/cpu_utilization_forecast.ipynb)
- [Alerting Server Response Time Forecasting](https://github.com/splunk/cisco-time-series-model/blob/main/1.0/notebooks/alerting_server_responsetime.ipynb)
- [Internet Traffic Forecasting](https://github.com/splunk/cisco-time-series-model/blob/main/1.0/notebooks/internet_traffic_forecast.ipynb)

<b>Notebooks contributed by:</b> Huaibo Zhao

## Citation
If you find Cisco Time Series Model useful for your research, please consider citing the associated technical report:
```
@misc{gou2025ciscotimeseriesmodel,
      title={Cisco Time Series Model Technical Report}, 
      author={Liang Gou and Archit Khare and Praneet Pabolu and Prachi Patel and Joseph Ross and Hercy Shen and Yuhan and Song and Jingze Sun and Kristal Curtis and Vedant Dharnidharka and Abhinav Mathur and Hao Yang},
      year={2025},
      eprint={2511.19841},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.19841}, 
}
```

## Authors:
- Liang Gou \*
- Archit Khare \*
- Praneet Pabolu \*
- Prachi Patel \*
- Joseph Ross \*
- Hercy Shen \*‡
- Yuhan (Ellen) Song \*
- Jingze Sun \*
- Kristal Curtis †
- Vedant Dharnidharka †
- Abhinav Mathur †
- Hao Yang †

\* These authors contributed equally to the core development of this work, listed alphabetically by last name. <br>
† These authors contributed equally to supporting and extending this work, listed alphabetically by last name.<br>
‡ Hercy Shen contributed to this work while an intern at Splunk.<br>

## License
This project is licensed under the Apache-2.0 License. See the [LICENSE](https://github.com/splunk/cisco-time-series-model/blob/main/LICENSE) file for more details.
