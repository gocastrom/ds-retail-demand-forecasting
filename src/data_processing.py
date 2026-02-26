import pandas as pd
import os

def load_data(items_path, stores_path, sales_path):
    """Carga los datos de items, tiendas y ventas."""
    print("Cargando datos...")
    items = pd.read_csv(items_path)
    stores = pd.read_csv(stores_path)
    sales = pd.read_csv(sales_path, parse_dates=['date'])
    print("Datos cargados exitosamente.")
    return items, stores, sales

def merge_data(sales, stores, items):
    """Une los dataframes de ventas, tiendas e items."""
    print("Uniendo dataframes...")
    df = sales.merge(stores, on='store_nbr', how='left')
    df = df.merge(items, on='item_nbr', how='left')
    print("Dataframes unidos exitosamente.")
    return df

def initial_cleaning(df):
    """Realiza una limpieza inicial y optimización de tipos de datos."""
    print("Realizando limpieza inicial...")
    df['store_nbr'] = df['store_nbr'].astype('int8')
    df['item_nbr'] = df['item_nbr'].astype('int32')
    df['unit_sales'] = df['unit_sales'].astype('float32')
    df['onpromotion'].fillna(False, inplace=True)
    df['onpromotion'] = df['onpromotion'].astype('bool')
    
    # Manejar valores negativos en ventas (no deberían existir)
    df['unit_sales'] = df['unit_sales'].apply(lambda u: u if u > 0 else 0)
    print("Limpieza completada.")
    return df

if __name__ == '__main__':
    # Rutas a los datos crudos
    ITEMS_PATH = 'data/raw/items.csv'
    STORES_PATH = 'data/raw/stores.csv'
    SALES_PATH = 'data/raw/train.csv' # Usaremos train.csv que tiene las ventas
    
    # Ruta de salida
    OUTPUT_DIR = 'data/processed'
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'full_data.parquet')
    
    # Crear directorio si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Ejecutar el pipeline de procesamiento
    items_df, stores_df, sales_df = load_data(ITEMS_PATH, STORES_PATH, SALES_PATH)
    merged_df = merge_data(sales_df, stores_df, items_df)
    cleaned_df = initial_cleaning(merged_df)
    
    # Guardar el resultado procesado
    print(f"Guardando datos procesados en '{OUTPUT_PATH}'...")
    cleaned_df.to_parquet(OUTPUT_PATH)
    print("¡Proceso de datos completado!")

