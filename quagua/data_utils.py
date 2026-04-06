"""
Data preprocessing utilities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_titanic_data(include_continuous=True):
    """
    Load and preprocess Titanic dataset
    
    Args:
        include_continuous: If True, include continuous numerical features (Age, Fare)
                          along with categorical/binary features. If False, only use
                          categorical/binary features.
    
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        feature_names: List of feature names
    """
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    
    # Select base features
    features = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare', 'Name', 'Cabin']
    target = 'Survived'
    
    # Create processed dataset
    df_processed = df[features + [target]].copy()
    
    # Handle missing values
    df_processed['Age'] = df_processed['Age'].fillna(df_processed['Age'].median())
    df_processed['Fare'] = df_processed['Fare'].fillna(df_processed['Fare'].median())
    df_processed['Embarked'] = df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0])
    
    # Extract Title from Name (Mr, Mrs, Miss, Master, etc.)
    df_processed['Title'] = df_processed['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col', 
                                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')
    
    # Family size features
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)
    
    # Has cabin flag
    df_processed['HasCabin'] = df_processed['Cabin'].notna().astype(int)
    
    # Encode categorical variables
    df_processed['Sex'] = LabelEncoder().fit_transform(df_processed['Sex'])
    df_processed['Embarked'] = LabelEncoder().fit_transform(df_processed['Embarked'])
    df_processed['Title'] = LabelEncoder().fit_transform(df_processed['Title'])
    
    # Convert SibSp and Parch to binary (0 vs >0)
    df_processed['SibSp_bin'] = (df_processed['SibSp'] > 0).astype(int)
    df_processed['Parch_bin'] = (df_processed['Parch'] > 0).astype(int)
    
    # Build final feature set
    if include_continuous:
        # Include both categorical/binary AND continuous numerical features
        # Categorical/binary: Pclass, Sex, Embarked, Title, SibSp_bin, Parch_bin, IsAlone, HasCabin
        # Continuous: Age, Fare, FamilySize (keep as continuous, not binned)
        final_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'SibSp_bin', 'Parch_bin', 
                         'IsAlone', 'HasCabin', 'Age', 'Fare', 'FamilySize']
        
        # Scale continuous features to have similar ranges
        # This helps encoders work better with mixed data types
        continuous_features = ['Age', 'Fare', 'FamilySize']
        categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'SibSp_bin', 'Parch_bin', 'IsAlone', 'HasCabin']
        
        # Scale continuous features using StandardScaler (mean=0, std=1)
        scaler = StandardScaler()
        df_processed[continuous_features] = scaler.fit_transform(df_processed[continuous_features])
        
        # Normalize categorical features to [0, 1] range for consistency
        for cat_feat in ['Pclass', 'Sex', 'Embarked', 'Title']:
            if df_processed[cat_feat].max() > 1:
                df_processed[cat_feat] = df_processed[cat_feat] / df_processed[cat_feat].max()
    else:
        # Original: only categorical/binary features
        # Bin Age into 3 categories (child, adult, senior)
        df_processed['Age_bin'] = pd.cut(df_processed['Age'], 
                                         bins=[0, 18, 60, 100], 
                                         labels=[0, 1, 2]).astype(int)
        final_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'SibSp_bin', 'Parch_bin', 
                         'IsAlone', 'HasCabin', 'Age_bin']
    
    X = df_processed[final_features].values
    y = df_processed[target].values
    
    return X, y, final_features


def load_adult_census_data():
    """
    Load and preprocess Adult Census Income dataset (UCI ML Repository)
    
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array, >50K = 1, <=50K = 0)
        feature_names: List of feature names
    """
    # Try to load from UCI or local file
    try:
        # Try UCI URL first
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        df = pd.read_csv(url, header=None, skipinitialspace=True, na_values='?')
    except:
        # Fallback: try local file or alternative URL
        try:
            url = 'https://raw.githubusercontent.com/selva86/datasets/master/adult.csv'
            df = pd.read_csv(url, na_values='?')
        except:
            raise ValueError("Could not load Adult Census dataset. Please download manually.")
    
    # Column names
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    if len(df.columns) == len(columns):
        df.columns = columns
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Target: income >50K = 1, <=50K = 0
    if 'income' in df.columns:
        df['target'] = (df['income'].str.strip() == '>50K').astype(int)
    else:
        # If target is in last column
        df['target'] = (df.iloc[:, -1].str.strip() == '>50K').astype(int)
    
    # Select features (exclude fnlwgt as it's a sampling weight, not a feature)
    feature_cols = ['age', 'workclass', 'education-num', 'marital-status', 
                    'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    
    # Keep only available columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    df_processed = df[feature_cols + ['target']].copy()
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if 'target' in categorical_cols:
        categorical_cols.remove('target')
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Scale numerical features
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numerical_cols:
        numerical_cols.remove('target')
    
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    # Normalize categorical features to [0, 1]
    for col in categorical_cols:
        if df_processed[col].max() > 1:
            df_processed[col] = df_processed[col] / df_processed[col].max()
    
    X = df_processed[feature_cols].values
    y = df_processed['target'].values
    
    return X, y, feature_cols


def load_credit_card_data():
    """
    Load and preprocess Credit Card Default dataset (UCI ML Repository)
    This is a real-world use case for financial privacy.
    
    Private features include:
    - Age: Personal demographic information
    - Bill amounts: Financial activity patterns
    - Payment amounts: Spending behavior
    - Credit limit: Financial status
    
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array, default = 1, no default = 0)
        feature_names: List of feature names
    """
    try:
        # Try UCI URL
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls'
        df = pd.read_excel(url, header=1, skiprows=0)
    except:
        try:
            # Alternative URL
            url = 'https://raw.githubusercontent.com/plotly/datasets/master/default_of_credit_card_clients.csv'
            df = pd.read_csv(url)
        except:
            raise ValueError("Could not load Credit Card dataset. Please download manually from UCI ML Repository.")
    
    # Remove ID column if present
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Target is last column (default payment next month)
    target_col = df.columns[-1]
    y = (df[target_col] == 1).astype(int).values
    
    # Features: exclude target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, feature_cols


def load_dataset(dataset_name):
    """
    Unified dataset loader
    
    Args:
        dataset_name: 'Titanic', 'Adult', or 'CreditCard'
    
    Returns:
        X, y, feature_names
    """
    if dataset_name.lower() == 'titanic':
        return load_titanic_data(include_continuous=True)
    elif dataset_name.lower() == 'adult':
        return load_adult_census_data()
    elif dataset_name.lower() == 'creditcard':
        return load_credit_card_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: 'Titanic', 'Adult', 'CreditCard'")
