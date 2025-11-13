#  HybGANN: Hybrid Genetic Algorithm Neural Network for Balanced Diabetes Prediction

This repository contains all Jupyter notebooks, models, and analysis scripts used for training, balancing, and evaluating the **HybGANN** framework — a hybrid deep learning model designed to improve diabetes prediction performance under imbalanced data conditions.

The project builds upon the **BRFSS 2015 Diabetes Health Indicators Dataset** from Kaggle and applies multiple **GAN-based balancing techniques**, comparative **ANN baselines**, and **interpretability analysis** using **SHAP** and **Jensen–Shannon Divergence (JSD)** metrics.

---

##  Datasets

All datasets used in this project are sourced from Kaggle:  
🔗 [Diabetes Health Indicators Dataset (BRFSS 2015)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_5050split_health_indicators_BRFSS2015.csv)

| Dataset Name | Description | Size | Usage |
|---------------|-------------|-------|--------|
| `Binarydiabetes_binary_health_indicators_BRFSS2015.csv` | Original **imbalanced** dataset | 22.74 MB | Used for baseline experiments and GAN-based balancing |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | Naturally **balanced** version (50/50 split) | ≈22 MB | Used for control and comparison |
| `CTGAN_balanced.csv`, `WGAN-GP_balanced.csv`, `TabFairGAN_balanced.csv` | Synthetic balanced datasets generated via GANs | varies | Used for comparative evaluation of balancing approaches |

---

## ⚙️ Experimental Design

###  Step 1. Data Balancing via GANs
Three different GAN architectures were implemented independently to balance the imbalanced diabetes dataset:
1. **CTGAN** – Conditional Tabular GAN  
2. **WGAN-GP** – Wasserstein GAN with Gradient Penalty  
3. **TabFairGAN** – Fairness-oriented tabular data generator  

Each GAN model produces a balanced dataset version saved separately, and its training stability and diversity were quantified using:
- **Jensen–Shannon Divergence (JSD)** – to assess distribution similarity  
- **SHAP analysis** – to interpret feature importance and fairness dynamics  

The code and results for each GAN are available in separate notebooks:
notebooks/
├── CTGAN_balancing.ipynb
├── WGAN_GP_balancing.ipynb
└── TabFairGAN_balancing.ipynb

markdown
Copy code

---

###  Step 2. HybGANN Evaluation

The **HybGANN model** (Hybrid Genetic Algorithm Neural Network) was tested on:
1. The **original imbalanced** dataset  
2. The **naturally balanced** dataset (50/50 split)  
3. The **GAN-balanced** datasets (CTGAN, WGAN-GP, TabFairGAN)

Performance metrics include:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**
- **Training efficiency** and **model robustness**
- **Feature interpretability** via SHAP  

All experimental notebooks are stored in:
notebooks/
├── HybGANN_original.ipynb
├── HybGANN_natural_5050.ipynb
├── HybGANN_CTGAN.ipynb
├── HybGANN_WGANGP.ipynb
└── HybGANN_TabFairGAN.ipynb

yaml
Copy code

---

###  Step 3. Baseline Comparison with ANN

A baseline **Artificial Neural Network (ANN)** model was also trained and tested on the **WGAN-GP balanced dataset**, since this version achieved the highest performance in HybGANN experiments.

This enables direct comparison between the hybrid model and a conventional ANN trained on the same balanced data.

Notebook:
notebooks/
└── ANN_baseline_WGANGP.ipynb
