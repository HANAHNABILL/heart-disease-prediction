# ❤️ Heart Disease Prediction System ❤️

A comprehensive machine learning-based web application for predicting heart disease risk using clinical parameters with an intuitive Streamlit interface.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-00A98F?style=for-the-badge)

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Deployment](#-deployment)

## ✨ Features:

###  Core Functionality
- **Accurate Predictions**: Machine learning model with 85-90% accuracy
- **Clinical Safety**: Built-in validation rules for edge cases
- **Real-time Analysis**: Instant risk assessment with clinical context

### User Experience
- **Modern UI**: Clean, healthcare-focused design with dark/light mode
- **Interactive Dashboard**: Comprehensive data visualization
- **Responsive Design**: Works on desktop and mobile devices

### Advanced Features
- **Multi-page Navigation**: Home, Prediction, Analysis, and Insights pages
- **Clinical Overrides**: Intelligent risk adjustment based on medical patterns
- **Data Visualization**: PCA analysis, correlation matrices, clustering insights

## 📁 Project Structure

```
Heart_Disease_Project/
├── kata/
│   └── heart_disease.csv              # Original dataset
├── notebooks/
│   ├── 1_data_preprocessing.ipynb     # Data cleaning & preparation
│   ├── 2_pca_analysis.ipynb           # PCA analysis
│   ├── 3_feature_selection.ipynb      # Feature selection
│   ├── 4_supervised_learning.ipynb    # Model training & optimization
│   ├── 5_unsupervised_learning.ipynb  # Clustering analysis
│   └── 6_hyperparameter_tuning.ipynb  # Hyperparameter optimization
├── models/
│   ├──  final_model.pkl               # Trained machine learning model
│ 
├── ui/
│   └── app.py                         # Streamlit application
├── deployment/
│   └── ngrok_setup.txt                # Deployment instructions
├── results/
│   └── evaluation_metrics.txt         # Model evaluation results
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction/Heart_Disease_Project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Run the Application
```bash
streamlit run ui/app.py
```

The application will open in your default browser at `http://localhost:8501`

### Run the Analysis Notebooks
Execute the notebooks in numerical order:
1. `1_data_preprocessing.ipynb`
2. `2_pca_analysis.ipynb`
3. `3_feature_selection.ipynb`
4. `4_supervised_learning.ipynb`
5. `5_unsupervised_learning.ipynb`
6. `6_hyperparameter_tuning.ipynb`

## 💻 Usage

### 1. Patient Risk Assessment
1. Launch the Streamlit app
2. Navigate to the "Prediction" page
3. Enter patient clinical parameters
4. Click "Assess Cardiovascular Risk"
5. View detailed risk analysis with clinical recommendations

### 2. Data Analysis
- Explore the "Analysis" page for dataset insights
- View feature distributions and correlations
- Perform PCA and clustering analysis

### 3. Model Insights
- Check the "Insights" page for feature importance
- Understand model decision factors
- Review clinical safety rules

## 🤖 Model Details

### Algorithm
- **Primary Model**: Optimized Random Forest classifier
- **Accuracy**: 90.16% on test data
- **Training Data**: 300+ patient records
- **Features**: 13 clinical parameters

### Performance Metrics
- **Accuracy**: 90.16%
- **F1-Score**: 89.66%
- **AUC Score**: 95.78%

### Clinical Parameters
- Demographics: Age, Sex
- Vital Signs: Resting BP, Max Heart Rate
- Blood Tests: Cholesterol, Fasting Blood Sugar
- Cardiac Metrics: Chest Pain Type, ECG Results, Exercise-induced Angina

## 🌐 Deployment

### Local Deployment with Ngrok

1. **Install Ngrok**
```bash
npm install -g ngrok
```

2. **Run the Application**
```bash
streamlit run ui/app.py --server.port 8501
```

3. **Expose to Web**
```bash
ngrok http 8501
```

4. **Share the generated URL**
   My Publicly accessible Streamlit app via Ngrok is: [https://contrite-lachlan-unbronzed.ngrok-free.dev](https://contrite-lachlan-unbronzed.ngrok-free.dev/)

### Cloud Deployment Options
- **Streamlit Sharing**: Push to GitHub and deploy via share.streamlit.io
- **Heroku**: Use the provided Procfile and setup.sh
- **AWS/Azure**: Containerize with Docker

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Medical Disclaimer

> **Important**: This application is for educational and research purposes only. It is not intended for clinical use or medical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

<div align="center">

**Made with ❤️ for better healthcare outcomes**

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/heart-disease-prediction&type=Date)](https://star-history.com/#yourusername/heart-disease-prediction&Date)

</div>

---
