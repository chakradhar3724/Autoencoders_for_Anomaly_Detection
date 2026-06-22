# Autoencoders for Anomaly Detection in Machine Temperature Data

This project builds an anomaly detection pipeline for industrial machine temperature readings using autoencoders. The goal is to learn the normal behavior of a machine from time-series windows and flag abnormal operating patterns using reconstruction error.

The project follows a model-development arc:

1. **Understand the signal** using exploratory analysis of machine temperature readings.
2. **Create anomaly labels** using a proxy threshold because the raw CSV does not provide ground-truth labels.
3. **Convert the time series into windows** so each model learns short-term temporal behavior.
4. **Train three autoencoder architectures** with increasing complexity.
5. **Compare performance and complexity** to choose the best model.
6. **Tune and evaluate the final model** using reconstruction-error thresholds, F1-score, and ROC-AUC.

---

## Dataset

The project uses the `machine_temperature_system_failure.csv` dataset from the Numenta Anomaly Benchmark (NAB) collection.

Each row contains:

| Column | Description |
|---|---|
| `timestamp` | Time of the temperature reading |
| `value` | Machine temperature measurement |

Dataset summary from the notebook:

| Property | Value |
|---|---:|
| Number of records | 22,695 |
| Number of features | 1 temperature value |
| Time interval | 5 minutes |
| Date range | 2013-12-02 to 2014-02-19 |
| Missing values | 0 |
| Mean temperature | 85.93 |
| Minimum temperature | 2.08 |
| Maximum temperature | 108.51 |

Since the provided CSV does not include explicit anomaly labels, the notebook creates proxy labels using the 93rd percentile of the temperature value:

```python
proxy_threshold = df["value"].quantile(0.93)
df["anomaly"] = (df["value"] > proxy_threshold).astype(int)
```

This creates the following class distribution:

| Class | Meaning | Count |
|---|---|---:|
| 0 | Normal | 21,106 |
| 1 | Proxy anomaly | 1,589 |

---

## Problem Statement

Industrial machines often show abnormal behavior before complete failure. This project treats anomaly detection as a reconstruction problem:

- The autoencoder is trained mostly on normal machine behavior.
- During testing, the model tries to reconstruct each input sequence.
- Normal windows should reconstruct well and have low reconstruction error.
- Anomalous windows should reconstruct poorly and have high reconstruction error.
- A threshold on reconstruction error is used to classify a window as normal or anomalous.

---

## Methodology

### 1. Data preprocessing

The notebook performs the following steps:

- Loads the machine temperature CSV.
- Converts `timestamp` to datetime format.
- Sorts the data chronologically.
- Checks for missing values.
- Normalizes the temperature values using `MinMaxScaler`.
- Splits the data chronologically into train, validation, and test sets.

Split used in the notebook:

| Split | Number of points |
|---|---:|
| Train | 15,886 |
| Validation | 3,404 |
| Test | 3,405 |

### 2. Window creation

The normalized time series is converted into sliding windows of length 30:

```python
SEQ_LEN = 30
```

Each window contains 30 consecutive readings. Since the readings are collected every 5 minutes, one window represents approximately 150 minutes of machine behavior.

Window-level labels are created as follows:

- If any point inside a window is anomalous, the full window is labeled anomalous.
- Otherwise, the window is labeled normal.

Window distribution:

| Split | Normal windows | Anomalous windows |
|---|---:|---:|
| Train | 13,970 | 1,886 |
| Validation | 3,100 | 274 |
| Test | 2,552 | 823 |

To avoid teaching the autoencoder to reconstruct anomalies, only normal training windows are used for model training.

---

## Models

Three autoencoder architectures were implemented and compared.

### Model 1: Dense Autoencoder

The first model is a simple fully connected autoencoder.

Architecture:

```text
Input: 30
Encoder: Linear(30 -> 16) -> ReLU -> Linear(16 -> 8) -> ReLU
Decoder: Linear(8 -> 16) -> ReLU -> Linear(16 -> 30) -> Sigmoid
```

This model is lightweight and directly reconstructs each 30-step window as a vector.

### Model 2: LSTM Autoencoder

The second model uses an LSTM encoder-decoder structure to capture sequential behavior.

Architecture:

```text
Input: sequence length 30, feature size 1
Encoder: LSTM(input_size=1, hidden_size=64)
Bottleneck: Linear(64 -> 32)
Decoder: LSTM(input_size=32, hidden_size=64)
Output: Linear(64 -> 1) -> Sigmoid
```

This model has more temporal modeling capacity than Model 1, but it is also significantly heavier.

### Model 3: Deeper LSTM Autoencoder

The third model increases the complexity of the LSTM architecture.

Architecture:

```text
Input: sequence length 30, feature size 1
Encoder: 2-layer LSTM(input_size=1, hidden_size=64)
Bottleneck: Linear(64 -> 16) -> Dropout(0.1)
Decoder: 2-layer LSTM(input_size=16, hidden_size=64)
Output: Linear(64 -> 1) -> Sigmoid
```

This model was tested to see whether a deeper recurrent model could improve anomaly detection, but the results showed that the extra complexity did not help on this dataset.

---

## Model Comparison

| Architecture | F1-score | ROC-AUC | Training Time / Epoch | Trainable Parameters |
|---|---:|---:|---:|---:|
| BestDenseAutoencoder / Model 1 | 0.7526 | 0.8331 | 0.9199 sec | 3,054 |
| Model 2: LSTM Autoencoder | 0.7130 | 0.8091 | 1.6785 sec | 44,385 |
| Model 3: Deeper LSTM Autoencoder | 0.7044 | 0.8062 | 2.0687 sec | 105,809 |

The dense autoencoder was selected as the final model because it achieved the best ROC-AUC and F1-score while using far fewer parameters than the LSTM models.

The important lesson from the model arc is that a more complex sequence model was not automatically better. For this dataset, the simpler dense autoencoder captured the normal temperature pattern more effectively and generalized better.

---

## Final Model

The final selected model is a tuned dense autoencoder.

Architecture:

```text
Input dimension: 30
Encoder: Linear(30 -> 32) -> ReLU -> Dropout(0.0) -> Linear(32 -> 16) -> ReLU
Bottleneck: 16-dimensional latent representation
Decoder: Linear(16 -> 32) -> ReLU -> Dropout(0.0) -> Linear(32 -> 30) -> Sigmoid
```

Training configuration:

| Hyperparameter | Value |
|---|---:|
| Sequence length | 30 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Epochs | 16 |
| Dropout | 0.0 |
| Bottleneck size | 16 |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Trainable parameters | 3,054 |

Training losses:

| Metric | Value |
|---|---:|
| Final training reconstruction loss | 0.000790 |
| Final validation reconstruction loss | 0.000972 |
| Test reconstruction loss | 0.001281 |

---

## Threshold Selection

The model produces a reconstruction error for each window. A threshold is then selected to convert reconstruction error into a binary prediction.

The notebook evaluates thresholds from the 70th to 99th percentile of training reconstruction errors and chooses the threshold that maximizes validation F1-score.

Best threshold from the notebook:

| Metric | Value |
|---|---:|
| Best threshold | 0.001125 |
| Validation F1-score | 0.4301 |
| Validation ROC-AUC | 0.8443 |

Lower thresholds catch more anomalies but increase false positives. Higher thresholds reduce false positives but may miss failures. For industrial failure detection, high recall is especially important because missing a true failure can be more costly than investigating a false alarm.

---

## Final Test Results

Final evaluation on the test set:

| Metric | Value |
|---|---:|
| Precision | 0.6311 |
| Recall | 0.9417 |
| F1-score | 0.7557 |
| ROC-AUC | 0.8331 |
| Accuracy | 0.85 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Normal | 0.98 | 0.82 | 0.89 | 2,552 |
| Anomaly | 0.63 | 0.94 | 0.76 | 823 |

The model catches about 94% of anomalous windows, which is useful for failure monitoring. The main tradeoff is precision: some normal windows are flagged as anomalous. This is expected because the sliding-window labeling strategy marks a full 30-step window as anomalous even when only part of the window contains abnormal readings.

---

## Key Visualizations in the Notebook

The notebook includes the following visual analysis:

- Raw temperature time-series plot
- Rolling mean plot
- Rolling standard deviation plot
- Training vs validation loss curves for each model
- ROC curve for the final model
- Reconstruction error distribution on the test set
- Threshold tuning table across reconstruction-error percentiles

These plots help explain both the behavior of the data and the reason behind the final model selection.

---

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch torchinfo
```

### 4. Add the dataset

Place the dataset file in the repository:

```text
machine_temperature_system_failure.csv
```

If you keep the CSV in the project root, update the notebook loading cell to:

```python
df = pd.read_csv("machine_temperature_system_failure.csv")
```

The original notebook contains a Colab-specific path. For a GitHub repo, using a relative path is recommended.

### 5. Run the notebook

Open the notebook:

```bash
jupyter notebook Autoencoders_for_Anomaly_Detection.ipynb
```

Then run all cells from top to bottom.

The notebook trains the models, saves the best dense autoencoder weights, tunes the anomaly threshold, and prints the final evaluation metrics.

---

## Limitations

- The anomaly labels are proxy labels based on a temperature percentile, not official ground-truth labels from the CSV.
- The threshold selection strongly affects precision and recall.
- Sliding-window labeling can inflate the number of anomaly windows near anomaly boundaries.
- The dataset has only one feature, so the model cannot use other machine signals such as pressure, vibration, or load.
- The current project is notebook-based; converting the workflow into Python scripts would make the repository easier to reuse and deploy.

---

## Future Improvements

Potential next steps:

- Use official NAB anomaly windows instead of percentile-based proxy labels.
- Add command-line scripts for training and inference.
- Save plots and metrics automatically into a `results/` folder.
- Experiment with Conv1D autoencoders or variational autoencoders.
- Compare against classical anomaly detection methods such as Isolation Forest, One-Class SVM, and statistical control charts.
- Add a simple inference function that accepts new temperature readings and returns anomaly scores.
- Create a small dashboard or Gradio app for interactive anomaly detection.

