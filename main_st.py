import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import LearningCurveDisplay
import io

def main():
    st.title("Modelo de Aprendizaje Automático para la Calidad del Agua")

    # 1. Cargar Shapefile
    st.header("1. Cargar Shapefile para entrenar el modelo")
    shapefile = st.file_uploader("Sube tu archivo shapefile (.shp, .shx, .dbf, .prj)", type=["shp", "shx", "dbf", "prj"], accept_multiple_files=True)

    if shapefile:
        # Lógica para manejar los múltiples archivos del shapefile
        # Streamlit carga los archivos como objetos individuales, necesitamos guardarlos temporalmente para que geopandas los pueda leer.
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shapefile:
                with open(os.path.join(tmpdir, f.name), "wb") as buffer:
                    buffer.write(f.getvalue())
            
            # Buscamos el archivo .shp para abrirlo
            shp_path = [os.path.join(tmpdir, f.name) for f in shapefile if f.name.endswith('.shp')][0]
            
            gdf = gpd.read_file(shp_path)

            st.write("Datos del Shapefile cargado:")
            st.dataframe(gdf.head())

            # 2. Análisis Exploratorio de Datos
            st.header("2. Análisis Exploratorio de Datos")
            
            # Mapa de calor de correlación
            st.subheader("Mapa de calor de correlación")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(gdf[["Blue","Green","Red","Red_Edge","NIR","NDWI","NDVI","Turbidez","Nitratos","Fosfatos"]].corr(),
                        annot=True, cmap='Blues', fmt='.2f', ax=ax)
            st.pyplot(fig)

            # Gráfico de pares
            st.subheader("Gráfico de pares")
            st.info("Este gráfico puede tardar un momento en generarse.")
            fig2 = sns.pairplot(gdf[["Blue","Green","Red","Red_Edge","NIR","NDWI","NDVI","Turbidez","Nitratos","Fosfatos"]], diag_kind='kde')
            st.pyplot(fig2)

            # 3. Selección de Parámetro y Modelado
            st.header("3. Selección de Parámetro y Entrenamiento del Modelo")
            selected_value = st.selectbox(
                'Selecciona el Parámetro de Calidad de Agua:',
                ('Turbidez', 'Nitratos', 'Fosfatos')
            )

            if selected_value:
                st.write(f"Parámetro seleccionado: {selected_value}")

                # Extraer variables
                X = gdf.drop(columns=['ID', 'Turbidez', 'Nitratos', 'Fosfatos', 'geometry'])
                y = gdf[selected_value]

                # División de datos
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
                
                # Estandarización
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Selección de características con Lasso
                st.subheader("Selección de Características con Regresión Lasso")
                lasso = Lasso(alpha=1)
                lasso.fit(X_train_scaled, y_train)
                lasso_coefficients = pd.Series(lasso.coef_, index=X_train.columns).sort_values()
                
                fig_lasso, ax_lasso = plt.subplots(figsize=(10, 6))
                sns.barplot(x=lasso_coefficients.values, y=lasso_coefficients.index, palette="coolwarm", ax=ax_lasso)
                ax_lasso.set_xlabel("Valor del Coeficiente")
                ax_lasso.set_ylabel("Variables")
                ax_lasso.set_title("Coeficientes del Modelo de Regresión Lasso")
                ax_lasso.axvline(x=0, color='black', linestyle='--', linewidth=1)
                st.pyplot(fig_lasso)


                # Entrenamiento de modelos
                st.subheader("Entrenamiento y Validación de Modelos")

                # SVR
                with st.spinner('Entrenando SVR...'):
                    param_grid_svr = {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1], 'kernel': ['rbf', 'linear', 'poly']}
                    svr = SVR()
                    svr_cv = GridSearchCV(svr, param_grid_svr, cv=5, n_jobs=-1)
                    svr_cv.fit(X_train, y_train)
                    optimized_svr = SVR(**svr_cv.best_params_)
                    optimized_svr.fit(X_train, y_train)
                st.success(f"Mejores parámetros para SVR: {svr_cv.best_params_}")

                # GBR
                with st.spinner('Entrenando Gradient Boosting...'):
                    param_grid_gbr = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 4, 5]}
                    gbr = GradientBoostingRegressor(random_state=13)
                    gbr_cv = GridSearchCV(estimator=gbr, param_grid=param_grid_gbr, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                    gbr_cv.fit(X_train, y_train)
                    optimized_gbr = GradientBoostingRegressor(random_state=13, **gbr_cv.best_params_)
                    optimized_gbr.fit(X_train, y_train)
                st.success(f"Mejores parámetros para GBR: {gbr_cv.best_params_}")

                # RFR
                with st.spinner('Entrenando Random Forest...'):
                    param_grid_rfr = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
                    rfr = RandomForestRegressor(random_state=13)
                    rfr_cv = GridSearchCV(estimator=rfr, param_grid=param_grid_rfr, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
                    rfr_cv.fit(X_train, y_train)
                    optimized_rfr = RandomForestRegressor(random_state=13, **rfr_cv.best_params_, n_jobs=-1)
                    optimized_rfr.fit(X_train, y_train)
                st.success(f"Mejores parámetros para RFR: {rfr_cv.best_params_}")


                # Validación de modelos
                st.subheader("Curvas de Aprendizaje")
                fig_lc, axes = plt.subplots(1, 3, figsize=(20, 5))
                
                common_params = {
                    "X": X_train, "y": y_train, "cv": ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
                    "n_jobs": -1, "train_sizes": np.linspace(0.1, 1.0, 5)
                }

                LearningCurveDisplay.from_estimator(optimized_svr, **common_params, ax=axes[0])
                axes[0].set_title(f"SVR")
                LearningCurveDisplay.from_estimator(optimized_gbr, **common_params, ax=axes[1])
                axes[1].set_title(f"Gradient Boosting")
                LearningCurveDisplay.from_estimator(optimized_rfr, **common_params, ax=axes[2])
                axes[2].set_title(f"Random Forest")
                st.pyplot(fig_lc)

                # Métricas de evaluación
                st.subheader("Métricas de Evaluación en el Conjunto de Prueba")
                y_pred_svr = optimized_svr.predict(X_test)
                y_pred_gbr = optimized_gbr.predict(X_test)
                y_pred_rfr = optimized_rfr.predict(X_test)

                metrics = {
                    'SVR': {'MAE': mean_absolute_error(y_test, y_pred_svr), 'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr)), 'R2': r2_score(y_test, y_pred_svr)},
                    'Gradient Boosting': {'MAE': mean_absolute_error(y_test, y_pred_gbr), 'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gbr)), 'R2': r2_score(y_test, y_pred_gbr)},
                    'Random Forest': {'MAE': mean_absolute_error(y_test, y_pred_rfr), 'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rfr)), 'R2': r2_score(y_test, y_pred_rfr)}
                }
                st.table(pd.DataFrame(metrics))
                
                # 4. Cargar imagen multiespectral y predecir
                st.header("4. Cargar Imagen Multiespectral para Predicción")
                
                band1 = st.file_uploader("Banda 1 (Azul)", type=["tif"])
                band2 = st.file_uploader("Banda 2 (Verde)", type=["tif"])
                band4 = st.file_uploader("Banda 4 (Rojo)", type=["tif"])
                band5 = st.file_uploader("Banda 5 (Borde Rojo)", type=["tif"])
                band6 = st.file_uploader("Banda 6 (NIR)", type=["tif"])
                ndwi_file = st.file_uploader("Índice NDWI", type=["tif"])
                ndvi_file = st.file_uploader("Índice NDVI", type=["tif"])

                if all([band1, band2, band4, band5, band6, ndwi_file, ndvi_file]):
                    with st.spinner('Procesando y prediciendo en la imagen...'):
                        # Leer bandas
                        with rasterio.open(band1) as src:
                            data1 = src.read()
                            profile = src.profile
                        with rasterio.open(band2) as src: data2 = src.read()
                        with rasterio.open(band4) as src: data4 = src.read()
                        with rasterio.open(band5) as src: data5 = src.read()
                        with rasterio.open(band6) as src: data6 = src.read()
                        with rasterio.open(ndwi_file) as src: data7 = src.read()
                        with rasterio.open(ndvi_file) as src: data8 = src.read()

                        # Combinar y remodelar
                        combined_data = np.vstack((data1, data2, data4, data5, data6, data7, data8))
                        height, width = combined_data.shape[1:]
                        image_reshaped = combined_data.reshape(combined_data.shape[0], -1).T

                        # Estandarizar y predecir
                        predicted_image_svr = optimized_svr.predict(image_reshaped)
                        predicted_image_rfr = optimized_rfr.predict(image_reshaped)
                        predicted_scaled_gbr = optimized_gbr.predict(image_reshaped)

                        # Reconstruir
                        image_svr = predicted_image_svr.reshape(height, width)
                        image_rfr = predicted_image_rfr.reshape(height, width)
                        image_gbr = predicted_scaled_gbr.reshape(height, width)
                    
                    st.success("Predicción completada.")

                    # 5. Guardar Imagen Generada
                    st.header("5. Descargar Imágenes Generadas")
                    
                    # Función para convertir array a archivo en memoria
                    def array_to_geotiff_bytes(data, profile):
                        profile.update(dtype=rasterio.float32, count=1)
                        memfile = io.BytesIO()
                        with rasterio.open(memfile, 'w', **profile) as dst:
                            dst.write(data.astype(rasterio.float32), 1)
                        memfile.seek(0)
                        return memfile.getvalue()

                    st.download_button(
                        label=f"Descargar {selected_value}_SVR_P.tif",
                        data=array_to_geotiff_bytes(image_svr, profile),
                        file_name=f"{selected_value}_SVR_P.tif",
                        mime="image/tiff"
                    )
                    st.download_button(
                        label=f"Descargar {selected_value}_RFR_P.tif",
                        data=array_to_geotiff_bytes(image_rfr, profile),
                        file_name=f"{selected_value}_RFR_P.tif",
                        mime="image/tiff"
                    )
                    st.download_button(
                        label=f"Descargar {selected_value}_GBR_P.tif",
                        data=array_to_geotiff_bytes(image_gbr, profile),
                        file_name=f"{selected_value}_GBR_P.tif",
                        mime="image/tiff"
                    )

if __name__ == "__main__":
    main()