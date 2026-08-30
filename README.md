# Heavy Machinery Price Prediction

An end-to-end Machine Learning project and feature engineering pipeline for predicting heavy equipment resale and auction prices (e.g., excavators, bulldozers, tractors).

## 📌 Features & Highlights
- **Comprehensive Feature Engineering**: Asset age calculation, usage intensity metrics, categorical encoding, and domain-specific interaction terms.
- **Multi-Model Ensembling**: Implementations of LightGBM, XGBoost, CatBoost, and Random Forest models.
- **Advanced Pipelines**: Stacking and blending strategies (Out-of-Fold predictions) to minimize RMSLE / RMSE metrics.
- **Jupyter Notebooks**: Complete versioned generation scripts and Jupyter notebooks for reproducible experiments.

## 📁 Repository Structure
```text
├── src_pipeline.py            # Main data loading and preprocessing pipeline
├── model_lightgbm.py          # LightGBM model configuration and training
├── model_xgboost.py           # XGBoost model setup
├── model_catboost.py          # CatBoost model implementation
├── model_randomforest.py      # Random Forest baseline model
├── model_hybrid_stacking.py   # Hybrid stacking and blending logic
├── notebooks/                 # Versioned Jupyter Notebooks
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/priyanshuanubis/Heavy-Machinery-Price-Prediction.git
   cd Heavy-Machinery-Price-Prediction
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

- **Run Training Pipeline**:
  ```bash
  python src_pipeline.py
  ```
- **Train Baseline / Specific Model**:
  ```bash
  python model_lightgbm.py
  python model_catboost.py
  ```
- **Run Hybrid Stacking**:
  ```bash
  python model_hybrid_stacking.py
  ```

## 📝 License
Distributed under the MIT License.
