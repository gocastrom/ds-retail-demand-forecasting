# **Retail Demand Forecasting 🛒📈**

End-to-end Machine Learning pipeline to forecast product demand per SKU and store using public retail sales data.

This project simulates a real retail analytics use case: **predict daily unit sales 7 days ahead to improve inventory planning and reduce stockouts.**

---

## **🚀 Problema de Negocio**

Las operaciones de retail requieren pronósticos de demanda precisos a corto plazo para:

*   Reducir quiebres de stock.
*   Optimizar la reposición de inventario.
*   Bajar los costos de mantenimiento de inventario.
*   Mejorar la planificación de promociones.
*   Aumentar la disponibilidad de productos en góndola.

Pronosticamos:

> `unit_sales(t + 7)`
> por `(tienda, SKU)`

---

## **📦 Dataset**

Dataset público de retail (Kaggle – Favorita Grocery Sales).

**Granularidad:**

*   Tienda (`store_nbr`)
*   SKU (`item_nbr`)
*   Fecha (`date`)
*   Ventas en unidades (`unit_sales`)
*   Promociones (`onpromotion`)
*   Metadata (atributos de tienda e item)

**Escala:**

*   Millones de filas.
*   Miles de SKUs.
*   Cientos de tiendas.

**Acción Requerida:** Para replicar este proyecto, descarga los datos desde [este link de Kaggle](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) y colócalos en la carpeta `data/raw/`.

---

## **🧠 Enfoque (Approach)**

### **1. Ingeniería de Datos**

*   Carga de datos (`.csv`).
*   Unión de tablas (ventas + tiendas + items).
*   Limpieza inicial y conversión de tipos.
*   Guardado en formato eficiente (`.parquet`).

### **2. Ingeniería de Características (time-series)**

Por cada SKU y Tienda:

*   **Lags:** `lag_7`, `lag_14`, `lag_28` (ventas de semanas anteriores).
*   **Ventanas Móviles:** Media de ventas en los últimos 7 y 28 días.
*   **Características Temporales:** Día de la semana, mes.
*   **Flag de Promoción.**

### **3. Modelado**

*   **Baseline:** Modelo ingenuo (usar `lag_7` como predicción).
*   **Machine Learning:** `HistGradientBoostingRegressor` de Scikit-learn (rápido y eficiente).

### **4. Evaluación**

*   **Métrica:** MAE (Mean Absolute Error).
*   **Validación:** División temporal (entrenar con el pasado, probar con el futuro).

---

## **📊 Resultados (Ejemplo)**

| Modelo | MAE (Error Absoluto Medio) |
| :--- | :--- |
| Baseline (lag-7) | XX.XX |
| HGB Regressor | XX.XX |

> Un MAE más bajo significa un pronóstico más preciso.

