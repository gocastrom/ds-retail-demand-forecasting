import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os

def prepare_data_for_modeling(df_path):
    """Carga los datos y los divide en entrenamiento y prueba de forma temporal."""
    print("Cargando datos con características...")
    df = pd.read_parquet(df_path)
    
    # Definir el punto de corte temporal (ej: últimas 2 semanas para test)
    test_date_start = df['date'].max() - pd.DateOffset(days=14)
    
    train = df[df['date'] < test_date_start]
    test = df[df['date'] >= test_date_start]
    
    # Definir características y objetivo
    features = [
        'day_of_week', 'month', 'onpromotion', 
        'lag_7', 'lag_14', 'lag_28',
        'rolling_mean_7', 'rolling_mean_28'
    ]
    target = 'unit_sales'

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    print(f"Tamaño de entrenamiento: {X_train.shape}, Tamaño de prueba: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    """Entrena el modelo HistGradientBoostingRegressor."""
    print("Entrenando el modelo...")
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    print("Modelo entrenado exitosamente.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo usando el error absoluto medio (MAE)."""
    print("Evaluando el modelo...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE del modelo: {mae:.4f}")
    return mae

if __name__ == '__main__':
    DATA_PATH = 'data/processed/final_features.parquet'
    MODEL_DIR = 'models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'demand_forecasting_model.joblib')

    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test = prepare_data_for_modeling(DATA_PATH)
    
    # Baseline (predecir con las ventas de la semana pasada)
    baseline_mae = mean_absolute_error(y_test, X_test['lag_7'])
    
    # Modelo ML
    hgb_model = train_model(X_train, y_train)
    model_mae = evaluate_model(hgb_model, X_test, y_test)
    
    print("\n--- Resultados Finales ---")
    print(f"MAE del Baseline (lag 7): {baseline_mae:.4f}")
    print(f"MAE del HGB Regressor:    {model_mae:.4f}")

    # Guardar el modelo entrenado
    print(f"Guardando modelo en '{MODEL_PATH}'...")
    joblib.dump(hgb_model, MODEL_PATH)
    print("¡Proceso de entrenamiento completado!")
