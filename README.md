# 🚀 End-to-End ML Pipeline with DVC + MLflow

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-Versioning-9cf.svg)](https://dvc.org)

> A complete production-ready ML pipeline with data version control (DVC), experiment tracking (MLflow), model deployment (FastAPI), and CI/CD automation.

---

## 📊 Project Overview

This project demonstrates a **complete MLOps pipeline** for machine learning models, covering the entire lifecycle from data ingestion to deployment.

### 🎯 What This Pipeline Does

| Stage | Tool | Purpose |
|-------|------|---------|
| **Data Versioning** | DVC | Version control for datasets |
| **Experiment Tracking** | MLflow | Log parameters, metrics, and models |
| **Model Training** | Scikit-learn | Train Random Forest classifier |
| **Model Deployment** | FastAPI | REST API for predictions |
| **Containerization** | Docker | Reproducible deployment |
| **CI/CD** | GitHub Actions | Automated testing and validation |

### 📊 Dataset

**Wine Dataset** – 178 samples, 13 features, 3 classes
- Features: Alcohol, Malic acid, Ash, etc.
- Target: 3 wine cultivars
- Source: UCI Machine Learning Repository

---

## 🏗️ Architecture

┌─────────────────────────────────────────────────────────────────┐
│ ML PIPELINE ARCHITECTURE │
├─────────────────────────────────────────────────────────────────┤
│ │
│ Raw Data ──► Preprocess ──► Train ──► Evaluate ──► Deploy │
│ │ │ │ │ │ │
│ ▼ ▼ ▼ ▼ ▼ │
│ DVC Scaler MLflow Reports FastAPI │
│ Versioning Saved Tracking Saved API │
│ │
└─────────────────────────────────────────────────────────────────┘


---

## 🛠️ Tech Stack

### Core Technologies
| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Programming language |
| Scikit-learn | 1.3.0 | Model training |
| Pandas | 2.0.3 | Data manipulation |
| NumPy | 1.24.3 | Numerical computing |

### MLOps Tools
| Tool | Version | Purpose |
|------|---------|---------|
| DVC | 3.0.0 | Data version control |
| MLflow | 2.5.0 | Experiment tracking |
| FastAPI | 0.100.0 | Model deployment API |
| Uvicorn | 0.23.1 | ASGI server |

### DevOps Tools
| Tool | Purpose |
|------|---------|
| Docker | Containerization |
| GitHub Actions | CI/CD automation |
| Pytest | Unit testing |

---

## 📁 Project Structure
ml-pipeline-dvc-mlflow/
│
├── data/
│ ├── raw/ # Raw dataset (versioned with DVC)
│ │ └── wine.csv
│ ├── processed/ # Preprocessed data
│ └── external/ # External data sources
│
├── src/
│ ├── init.py
│ ├── data_loader.py # Load and save raw data
│ ├── preprocess.py # Feature scaling
│ ├── train.py # Model training with MLflow
│ ├── evaluate.py # Model evaluation
│ └── predict.py # Prediction functions
│
├── api/
│ ├── init.py
│ └── main.py # FastAPI application
│
├── models/
│ ├── model.pkl # Trained model
│ └── scaler.pkl # Fitted scaler
│
├── reports/
│ ├── metrics.json # Model metrics
│ ├── classification_report.csv
│ └── confusion_matrix.png
│
├── mlruns/ # MLflow experiment data
├── .dvc/ # DVC cache
├── .github/
│ └── workflows/
│ └── ci.yml # GitHub Actions CI/CD
│
├── tests/
│ ├── test_api.py
│ └── test_model.py
│
├── dvc.yaml # DVC pipeline definition
├── Dockerfile # Docker configuration
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md


---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Docker (optional)
- Virtual environment (recommended)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/younESCOO2003/ml-pipeline-dvc-mlflow.git
cd ml-pipeline-dvc-mlflow

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate