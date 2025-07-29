# RFM‑Based Customer Segmentation and Prediction Software

This project implements an end‑to‑end pipeline for customer segmentation based on RFM (Recency, Frequency, Monetary) analysis and then builds a machine‑learning model to predict a customer’s segment. A Streamlit app is also included for interactive exploration and live predictions.

---

## 🔍 Table of Contents

1. [Features](#-features)
2. [Repository Structure](#-repository-structure)
3. [Getting Started](#-getting-started)

   * [Prerequisites](#prerequisites)
   * [Installation](#installation)
4. [Usage](#-usage)

   * [1. RFM Analysis](#1-rfm-analysis)
   * [2. Clustering & Segmentation](#2-clustering--segmentation)
   * [3. Model Training & Evaluation](#3-model-training--evaluation)
   * [4. Prediction API / Streamlit App](#4-prediction-api--streamlit-app)
5. [Results & Visualization](#-results--visualization)
6. [Contributing](#-contributing)
7. [License](#-license)
8. [Contact](#-contact)

---

## 🔍 Features

* **RFM Analysis**: Compute Recency, Frequency, and Monetary metrics from transactional data.
* **Customer Segmentation**: Apply K‑Means clustering to segment customers into behavioral groups.
* **Predictive Modeling**: Train a Random Forest classifier to predict segment labels for new customers.
* **Interactive App**: A Streamlit interface for uploading new customer data and getting live segment predictions.
* **Visualizations**: Jupyter Notebook and static plots to explore RFM distributions and cluster characteristics.

---

## 🗂 Repository Structure

All key files and their direct links in the repository:

```
├── [.devcontainer/](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/tree/main/.devcontainer)
├── [.streamlit/](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/tree/main/.streamlit)
├── [Online_Retail_Train.zip](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/Online_Retail_Train.zip)
├── [RFM_Analysis.py](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/RFM_Analysis.py)
├── [KMeans_Model.py](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/KMeans_Model.py)
├── [RandomForest_Model.py](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/RandomForest_Model.py)
├── [predictor.py](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/predictor.py)
├── [Plots.ipynb](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/Plots.ipynb)
├── [rf.pkl](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/rf.pkl)
├── [confusion_matrix_rf.png](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/confusion_matrix_rf.png)
├── [newplot.png](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/newplot.png)
├── [requirements.txt](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/requirements.txt)
└── [README.md](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/README.md)  ← You are here
```

---

## 🔗 Deployed Application

Access the live app at: [Deployed Streamlit App](https://rfm-based-customer-segmentation-and-prediction-software.streamlit.app)

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Git

### Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software.git
   cd RFM-Based-Customer-Segmentation-and-Prediction-Software
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Linux / macOS
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. RFM Analysis

Generate the RFM table from your raw transactions CSV (e.g. `Online_Retail_Train.zip`):

```bash
python RFM_Analysis.py \
  --input_data path/to/transactions.csv \
  --output_rfm rfm_table.csv
```

This script outputs a CSV of customer RFM features and basic summary statistics.

### 2. Clustering & Segmentation

Run K‑Means on the computed RFM features to assign each customer to a segment:

```bash
python KMeans_Model.py \
  --rfm_data rfm_table.csv \
  --n_clusters 4 \
  --output_labels rfm_segments.csv
```

Adjust `--n_clusters` as needed for your business case.

### 3. Model Training & Evaluation

Train a Random Forest classifier to predict segment labels and save the model:

```bash
python RandomForest_Model.py \
  --features rfm_table.csv \
  --labels rfm_segments.csv \
  --output_model rf.pkl
```

Evaluation metrics and a confusion matrix are displayed and saved (`confusion_matrix_rf.png`).

### 4. Prediction API / Streamlit App

Use `predictor.py` in a production or batch setting:

```bash
python predictor.py \
  --model_path rf.pkl \
  --new_customers new_customers.csv \
  --predictions output_segments.csv
```

Or launch the Streamlit app for an interactive UI:

```bash
streamlit run predictor.py
```

---

## 📊 Results & Visualization

* **[Plots.ipynb](https://github.com/AshTron811/RFM-Based-Customer-Segmentation-and-Prediction-Software/blob/main/Plots.ipynb)** contains exploratory analysis of RFM distributions and cluster profiles.
* Check the `/plots` folder for sample output PNGs like `newplot.png`.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/xyz`)
3. Commit your changes (`git commit -m 'Add xyz'`)
4. Push to the branch (`git push origin feature/xyz`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 📬 Contact

**Ashutosh Sharma**

* GitHub: [@AshTron811](https://github.com/AshTron811)
* Email: [your.email@example.com](mailto:your.email@example.com)

Feel free to open an issue or reach out if you have any questions or suggestions!
