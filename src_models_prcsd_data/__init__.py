"""
ML Workflow Pipeline with Data Preprocessing, Model Training, and Testing

This module provides a machine learning pipeline that includes data loading, preprocessing, model training, saving, and testing. It supports Gradient Boosting, Decision Tree, and Random Forest classifiers.

Classes:
- DataPreprocessor: Abstract base class for data preprocessing.
- BasicPreprocessor: Implements basic data cleaning, feature conversion, and splitting.
- ModelHandler: Abstract base class for machine learning models.
- GradientBoostingModel: Concrete implementation of a Gradient Boosting Classifier.
- MLWorkflow: Manages the entire workflow including data handling, model training, and testing.

Dependencies:
- os, pandas, numpy, abc, sklearn, joblib

"""

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

class DataPreprocessor(ABC):
    """Abstract base class for data preprocessing that requires a preprocess method."""
    @abstractmethod
    def preprocess(self, df, target_col):
        pass

class BasicPreprocessor(DataPreprocessor):
    """Performs basic data cleaning, including handling duplicates, missing values, and converting categorical data to numerical codes."""
    def preprocess(self, df, target_col, selected_columns=None):
        df = df.drop_duplicates()
        if selected_columns:
            df = df[selected_columns]
        for col in df.select_dtypes(include=['object']).columns:
            converted = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].astype('category').cat.codes if converted.isna().any() else converted
        df = df.dropna()
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        return df.drop(columns=[target_col]), df[target_col]

class ModelHandler(ABC):
    """Abstract base class for machine learning models with train and predict methods."""
    @abstractmethod
    def train(self, X_train, y_train):
        pass
    @abstractmethod
    def predict(self, X_test):
        pass

class GradientBoostingModel(ModelHandler):
    """Gradient Boosting Classifier implementation using sklearn."""
    def __init__(self, random_state=18):
        self.model = GradientBoostingClassifier(random_state=random_state)
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self, X_test):
        return self.model.predict(X_test)

class MLWorkflow:
    """Manages the ML workflow including data loading, preprocessing, model training, saving, and testing."""
    def __init__(self, models_dir='models', model_path=None):
        self.models_dir = models_dir
        self.model_path = model_path
        os.makedirs(self.models_dir, exist_ok=True)

    def load_data(self, file_path):
        """Loads CSV data from a specified path and returns a DataFrame."""
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded: {file_path}, shape: {data.shape}")
            return data
        except Exception as e:
            raise ValueError(f"Data loading error: {e}")

    def get_next_model_name(self, model_name):
        """Generates a unique filename for the trained model based on its type."""
        model_ids = {"Gradient Boosting": "gb", "Decision Tree": "dt", "Random Forest": "rf"}
        model_id = model_ids.get(model_name, "model")
        existing = [f for f in os.listdir(self.models_dir) if f.startswith(f'{model_id}_model') and f.endswith('.pkl')]
        next_num = max([int(f.split('_')[2].split('.')[0]) for f in existing if f.split('_')[2].split('.')[0].isdigit()], default=0) + 1
        return f'{model_id}_model_{next_num}.pkl'

    def train_model(self, train_path, target_col, selected_columns=None, test_size=0.2, random_state=18, model_name="Gradient Boosting", model_params=None):
        """Trains the specified ML model using the provided dataset and parameters."""
        try:
            df = self.load_data(train_path)
            X, y = BasicPreprocessor().preprocess(df, target_col, selected_columns)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            model_params = model_params or {}
            models = {"Gradient Boosting": GradientBoostingClassifier, "Decision Tree": DecisionTreeClassifier, "Random Forest": RandomForestClassifier}
            model = models.get(model_name, lambda: (_ for _ in ()).throw(ValueError("Unsupported model")))(random_state=random_state, **model_params)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            self.model_path = os.path.join(self.models_dir, self.get_next_model_name(model_name))
            joblib.dump(model, self.model_path)
            print(f"Model saved at {self.model_path}")
            return acc
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

    def test_model(self, test_path, target_col, selected_columns=None):
        """Tests the most recent trained model on the provided dataset."""
        try:
            df = self.load_data(test_path)
            X_test, y_test = BasicPreprocessor().preprocess(df, target_col, selected_columns)
            models = sorted([f for f in os.listdir(self.models_dir) if f.endswith('.pkl')], key=lambda x: int(x.split('_')[2].split('.')[0]))
            if not models:
                raise FileNotFoundError("No trained models available.")
            model = joblib.load(os.path.join(self.models_dir, models[-1]))
            return accuracy_score(y_test, model.predict(X_test))
        except Exception as e:
            raise RuntimeError(f"Testing failed: {e}")