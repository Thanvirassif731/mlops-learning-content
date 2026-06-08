import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

DEFAULT_DATA_PATH = 'data/User1_Cleaned.csv'
DEFAULT_MODEL_DIR = 'models'


def parse_time(value):
    if pd.isna(value) or value == '-':
        return 0
    cleaned = str(value).replace('s', '')
    parts = cleaned.split('m ')
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]) * 60 + int(parts[1])
    return 0


def engineer_features(data):
    df_clean = data.copy()
    
    # Target 1: Survival (1 if Not Murdered AND Not Ejected, else 0)
    df_clean['Survived'] = ((df_clean['Murdered'] == 'No') & (df_clean['Ejected'] == 'No')).astype(int)
    
    # Target 2: Sabotages Fixed (handle N/A and hyphens)
    df_clean['Sabotages Fixed'] = pd.to_numeric(
        df_clean['Sabotages Fixed'].replace(['N/A', '-'], 0), errors='coerce'
    ).fillna(0)
    
    # Feature: Convert Game Length (e.g., '07m 04s') to total seconds
    def parse_time(t):
        if pd.isna(t) or t == '-': return 0
        parts = str(t).replace('s', '').split('m ')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0
        
    df_clean['Game Length Sec'] = df_clean['Game Length'].apply(parse_time)
    
    # Clean up numeric features with hyphens
    df_clean['Task Completed'] = pd.to_numeric(df_clean['Task Completed'].replace('-', 0), errors='coerce').fillna(0)
    df_clean['Imposter Kills'] = pd.to_numeric(df_clean['Imposter Kills'].replace('-', 0), errors='coerce').fillna(0)
    
    return df_clean


def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at: {path}")
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ['Task Completed', 'Imposter Kills', 'Game Length Sec']
    categorical_features = ['Team']
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )


def build_models():
    preprocessor = build_preprocessor()
    survive_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    sabotage_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    return survive_pipe, sabotage_pipe


def train_and_serialize(data_path: str = DEFAULT_DATA_PATH, model_dir: str = DEFAULT_MODEL_DIR):
    df = load_data(data_path)
    df_processed = engineer_features(df)

    X = df_processed[['Team', 'Task Completed', 'Imposter Kills', 'Game Length Sec']]
    y_survive = df_processed['Survived']
    y_sabotage = df_processed['Sabotages Fixed']

    survive_pipe, sabotage_pipe = build_models()

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_survive, test_size=0.2, random_state=42
    )
    survive_pipe.fit(X_train_s, y_train_s)

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_sabotage, test_size=0.2, random_state=42
    )
    sabotage_pipe.fit(X_train_b, y_train_b)

    os.makedirs(model_dir, exist_ok=True)
    survive_path = os.path.join(model_dir, 'survive_model.pkl')
    sabotage_path = os.path.join(model_dir, 'sabotage_model.pkl')
    joblib.dump(survive_pipe, survive_path)
    joblib.dump(sabotage_pipe, sabotage_path)

    print('Pipelines trained and serialized successfully!')
    return survive_pipe, sabotage_pipe


if __name__ == '__main__':
    train_and_serialize()

