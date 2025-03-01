# Machine Learning Pipeline Project

## 🚀 Project Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git

### 1. Clone Repository
```bash
# Clone the project
git clone https://github.com/Zark-ML/Diabetes-Dataset
cd Diabetes-Dataset
```

### 2. Virtual Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

## 🔧 Project Structure
```
├── Data_Analyse                      # Data analyse of 
├── src_models_prcsd_data/            # Folder in which we have saved processed datasets and source files
|         ├── train_processed.csv     # Processed train dataset
|         ├── test_processed.csv      # Processed test dataset
│         ├── __init__.py             # ML core logic
│         └── pipeline.py             # GUI implementation
├── .gitignore                        # Contains files to ignore
├── data/                             # Dataset storage (test/train and the raw data(csv))
├── requirements.txt                  # Required python libraries
└── README.md                         # Tutorial file (follow along)

```

## 💻 Application Workflow

### Data Preparation
1. Prepare CSV dataset with features and target column
2. Ensure data is clean and preprocessed

### Running the GUI
```bash
# Launch ML Pipeline GUI
python src/pipeline.py
```

### GUI Navigation
1. **Data Loading**
   - Click "Browse" 
   - Select input CSV file for training model
   - Choose target column
   - Select relevant features
   - Do preprocessing of the dataset by hitting "Process Train Data" button

2. **Model Configuration**
   - Select model type:
     * Gradient Boosting
     * Decision Tree
     * Random Forest
   - Adjust model hyperparameters
   - Click "Train Model"
   - See the result of your selected model (Accuracy)

3. **Model Evaluation**
   - Load test dataset
   - Process the dataset by clicking the corresponding button
   - Click "Test Model"
   - See the test results (Accuracy)

### Custom Prediction
- Input individual data points
- Click "Predict Custom Data Point" for single sample classification
- See whether it has Diabetes(Predicts Target column) or hasn't

## 🛠 Troubleshooting
- Verify Python version: `python --version`
- Check pip installation: `pip --version`
- Reinstall dependencies if errors occur: `pip install -r requirements.txt`

## 📋 Requirements
- pandas
- scikit-learn
- tkinter
- joblib

## 🆘 Support
Open GitHub issues for bugs or feature requests.

