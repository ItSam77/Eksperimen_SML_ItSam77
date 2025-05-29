import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Comprehensive data preprocessing function that handles all preprocessing steps.
    
    Args:
        df (pandas.DataFrame): Raw input DataFrame
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame ready for training
    """
    # Make a copy to avoid modifying the original data
    df = df.copy()
    
    # 1. Data Cleaning - Remove missing values and duplicates
    df = df.dropna().drop_duplicates()
    
    # 2. Remove constant features that don't add value
    # Drop columns that have the same value for all records
    constant_columns = ['EmployeeCount', 'Over18', 'StandardHours']
    df = df.drop(columns=constant_columns, errors='ignore')
    
    # 3. Drop EmployeeNumber as it's just an identifier
    df = df.drop(columns=['EmployeeNumber'], errors='ignore')
    
    # 4. Feature Engineering - Create LoyaltyRatio
    if 'YearsAtCompany' in df.columns and 'TotalWorkingYears' in df.columns:
        # Avoid division by zero
        df['LoyaltyRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    
    # 5. Handle Target Variable Encoding
    if 'Attrition' in df.columns:
        # Encode Attrition: Yes -> 1, No -> 0
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # 6. Encode Categorical Variables
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from categorical columns if it exists
    if 'Attrition' in categorical_columns:
        categorical_columns.remove('Attrition')
    
    # Apply Label Encoding to categorical variables
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # 7. Feature Scaling - Apply StandardScaler to numerical columns
    # Get numerical columns (excluding binary encoded columns and target)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target variable from scaling if it exists
    if 'Attrition' in numerical_cols:
        numerical_cols.remove('Attrition')
    
    # Apply StandardScaler to numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df
