# 📱 Telecom Churn Prediction App

A machine learning web application that predicts customer churn for telecom companies based on usage patterns and service characteristics.

## 🚀 Live Demo

**Try the app:** [https://telecom-churn-prediction-nywqznfccej4tprinc5b2s.streamlit.app/](https://telecom-churn-prediction-nywqznfccej4tprinc5b2s.streamlit.app/)

## 📋 Overview

This project uses a Random Forest machine learning model to predict whether a telecom customer is likely to churn (leave the service). The model analyzes various customer attributes including account information, service plans, and usage statistics to make accurate predictions.

## ✨ Features

- **Interactive Web Interface**: Built with Streamlit for easy-to-use predictions
- **Real-time Predictions**: Get instant churn probability predictions
- **Sample Data Viewer**: Explore the training dataset
- **Model Information**: Learn about the Random Forest classifier used
- **Customer Input Form**: Enter custom customer data for predictions

## 🛠️ Technologies Used

- **Python 3.13**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning model (Random Forest)
- **NumPy** - Numerical computations

## 📊 Model Details

The prediction model uses the following customer features:

### Account Information
- Account Length (days)
- International Plan (Yes/No)
- Voice Mail Plan (Yes/No)

### Usage Statistics
- Day Minutes/Calls/Charge
- Evening Minutes/Calls/Charge
- Night Minutes/Calls/Charge
- International Minutes/Calls/Charge
- Voice Mail Messages
- Customer Service Calls

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.10
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sumitbavaskar/Telecom-Churn-Prediction.git
cd Telecom-Churn-Prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
Telecom-Churn-Prediction/
├── app.py                          # Main Streamlit application
├── random_forest_model.pkl         # Trained ML model
├── telecom_churn_cleaned.csv       # Training dataset
├── requirements.txt                # Python dependencies
├── Telecom Project.ipynb          # Model training notebook
├── README.md                       # Project documentation
└── DEPLOYMENT.md                   # Deployment guide
```

## 💡 Usage

1. **Visit the Live App**: Go to the deployed application link
2. **Enter Customer Details**: Fill in the customer information in the sidebar
3. **Get Prediction**: Click to see the churn prediction and probability
4. **Explore Data**: Use the expandable sections to view sample data and model info

## 📈 Model Performance

The Random Forest classifier was trained on historical telecom customer data with features including:
- Customer demographics
- Service usage patterns
- Plan information
- Customer service interactions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Sumit Bavaskar**
- GitHub: [@sumitbavaskar](https://github.com/sumitbavaskar)

## 🙏 Acknowledgments

- Dataset: Telecom Customer Churn Dataset
- Framework: Streamlit for the amazing web app framework
- ML Library: Scikit-learn for powerful machine learning tools

---

**Made with ❤️ using Streamlit**
