# Streamlit Deployment Guide

## Deploying Your Telecom Churn Prediction App on Streamlit Cloud

This guide will help you deploy your Telecom Churn Prediction Streamlit app to the cloud.

### Prerequisites
- GitHub account
- All necessary files in your repository:
  - `app.py` (Streamlit application)
  - `requirements.txt` (Python dependencies)
  - `random_forest_model.pkl` (Trained model)
  - `telecom_churn_cleaned.csv` (Data file)

### Step 1: Prepare Your Repository

Make sure your repository has the following structure:
```
Telecom-Churn-Prediction/
├── app.py
├── requirements.txt
├── random_forest_model.pkl
├── telecom_churn_cleaned.csv
├── Telecom Project.ipynb
└── README.md
```

### Step 2: Deploy on Streamlit Cloud

1. **Visit Streamlit Cloud**
   - Go to [https://share.streamlit.io/](https://share.streamlit.io/)
   - Sign in with your GitHub account

2. **Create New App**
   - Click "New app" button
   - Select your repository: `sumitbavaskar/Telecom-Churn-Prediction`
   - Set branch to: `main`
   - Set main file path to: `app.py`

3. **Deploy**
   - Click "Deploy!" button
   - Wait for deployment to complete (usually takes 2-3 minutes)

4. **Access Your App**
   - Once deployed, you'll get a public URL like:
     `https://sumitbavaskar-telecom-churn-prediction-app.streamlit.app`

### Step 3: Test Your Deployed App

1. Open the deployed URL
2. Fill in customer information in the sidebar
3. Click "Predict Churn" to see predictions
4. Verify that the model loads correctly and predictions work

### Troubleshooting

**If deployment fails:**

1. **Check logs** in Streamlit Cloud dashboard for errors

2. **Common issues:**
   - Missing files: Ensure all files (model.pkl, data.csv) are in the repo
   - Dependency errors: Verify requirements.txt has correct versions
   - Memory issues: Large model files might cause issues (your model is ~2MB, which should be fine)

3. **Update dependencies** if needed:
   ```txt
   streamlit>=1.28.0
   pandas>=2.0.0
   scikit-learn>=1.3.0
   numpy>=1.24.0
   ```

### Local Testing Before Deployment

Test your app locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Updating Your Deployed App

To update your deployed app:
1. Make changes to your code
2. Commit and push to GitHub
3. Streamlit Cloud will automatically redeploy

### Additional Features

**Custom Domain (Optional)**
- You can set up a custom domain in Streamlit Cloud settings

**App Settings**
- Configure secrets for sensitive data
- Set up advanced settings in `.streamlit/config.toml`

**Analytics**
- View app usage statistics in Streamlit Cloud dashboard

### Support

For issues:
- Check [Streamlit Documentation](https://docs.streamlit.io/)
- Visit [Streamlit Community Forum](https://discuss.streamlit.io/)

---

## Quick Start Commands

```bash
# Clone repository
git clone https://github.com/sumitbavaskar/Telecom-Churn-Prediction.git
cd Telecom-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Project Information

- **Model**: Random Forest Classifier
- **Accuracy**: 93.29%
- **Features**: 80 (including usage patterns, service plans, demographics)
- **Technology Stack**: Python, Streamlit, Scikit-learn, Pandas

---

**Created by**: Sumit Bavaskar  
**Repository**: [Telecom-Churn-Prediction](https://github.com/sumitbavaskar/Telecom-Churn-Prediction)  
**Last Updated**: November 2025
