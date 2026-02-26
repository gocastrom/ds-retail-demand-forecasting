import pandas as pd
import os

def create_time_features(df):
    """Crea características basadas en la fecha."""
    print("Creando características de tiempo...")
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek.astype('int8')
    df['month'] = df['date'].dt.month.astype('int8')
    print("Características de tiempo creadas.")
    return df

def create_lag_features(df, lags):
    """Crea características de lag para las ventas."""
    print(f"Creando características de lag: {lags}...")
    df_sorted = df.sort_values(['store_nbr', 'item_nbr', 'date'])
    for lag in lags:
        df_sorted[f'lag_{lag}'] = df_sorted.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(lag)
    print("Características de lag creadas.")
    return df_sorted

def create_rolling_features(df, windows):
    """Crea características de ventana móvil (rolling window)."""
    print(f"Creando características de ventana móvil: {windows}...")
    df_sorted = df.sort_values(['store_nbr', 'item_nbr', 'date'])
    for window in windows:
        # El shift(1) es para no usar el dato del día actual en el cálculo
        df_sorted[f'rolling_mean_{window}'] = df_sorted.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(1).rolling(window).mean()
    print("Características de ventana móvil creadas.")
    return df_sorted

if __name__ == '__main__':
    INPUT_PATH = 'data/processed/full_data.parquet'
    OUTPUT_PATH = 'data/processed/final_features.parquet'
    
    print("Cargando datos pre-procesados...")
    df = pd.read_parquet(INPUT_PATH)
    
    # Pipeline de creación de características
    df = create_time_features(df)
    df = create_lag_features(df, lags=[7, 14, 28])
    df = create_rolling_features(df, windows=[7, 28])
    
    # Rellenar NaNs generados por las lags/rolling
    df.fillna(0, inplace=True)

    # Guardar el resultado final
    print(f"Guardando datos con características en '{OUTPUT_PATH}'...")
    df.to_parquet(OUTPUT_PATH)
    print("¡Proceso de creación de características completado!")
