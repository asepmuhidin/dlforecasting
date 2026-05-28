# DLForecasting

DLForecasting adalah package Python untuk forecasting time series menggunakan Deep Learning berbasis:

* RNN
* LSTM
* GRU

Package ini dibuat menggunakan PyTorch dan mendukung:

* data preprocessing
* scaling
* sequence/windowing
* training model
* evaluasi model
* penyimpanan model

---

# Instalasi

```bash
pip install dlforecasting
```

atau dari source:

```bash
git clone https://github.com/username/dlforecasting.git

cd dlforecasting

pip install -r requirements.txt
```

---

# Dependencies

* torch
* numpy
* pandas
* scikit-learn
* tqdm

---

# Contoh Penggunaan

```python
import pandas as pd

from dlforecasting import dlforecasting

# load dataset
df = pd.read_csv("bbca.csv")

series = df[['Close']]

# training
result = dlforecasting(
    series=series,

    model_type='LSTM',

    hidden_size=64,

    epoch_size=50,

    window_size=30
)

# metrics
print(result['metrics'])
```

---

# Parameter

| Parameter   | Deskripsi                |
| ----------- | ------------------------ |
| train_size  | Persentase data training |
| input_size  | Jumlah fitur input       |
| hidden_size | Jumlah neuron hidden     |
| num_layers  | Jumlah layer RNN         |
| model_type  | RNN / LSTM / GRU         |
| epoch_size  | Jumlah epoch training    |
| window_size | Panjang sequence         |
| batch_size  | Ukuran batch             |
| lr          | Learning rate            |
| seed        | Random seed              |

---

# Output

Function `dlforecasting()` mengembalikan dictionary:

```python
{
    'model': model,
    'history': history,
    'metrics': metrics,
    'config': config,
    'scaler': scaler
}
```

---

# Metrics

Evaluasi yang tersedia:

* RMSE
* MAE
* MAPE

---

# Save Model

Model otomatis disimpan dalam format:

```text
lstm_window30_hidden64.pth
```

---

# Author

Asep Muhidin

---

# License

MIT License
