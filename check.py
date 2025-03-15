import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import streamlit as st

# Judul aplikasi Streamlit
st.title('Model Peramalan Time Series')

# Unggah file CSV
uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])

if uploaded_file is not None:
    # Memuat dataset
    df = pd.read_csv(uploaded_file)

    # Mengonversi kolom 'Tanggal' ke format datetime dan mengatur sebagai indeks
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.set_index('Tanggal', inplace=True)

    # Menampilkan data
    st.write("Pratinjau Data:")
    st.dataframe(df)

    # Sidebar untuk pemilihan model dan variabel throughput
    st.sidebar.header("Pengaturan")
    
    # Pilihan variabel throughput di sidebar
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_throughput = st.sidebar.selectbox(
        "Pilih variabel throughput", 
        options=numeric_columns,
        index=0 if 'throughput' in numeric_columns else 0
    )
    
    st.sidebar.header("Pemilihan Model")
    use_arima = st.sidebar.checkbox("ARIMA", value=True)
    use_sarimax = st.sidebar.checkbox("SARIMAX", value=True)

    # Membagi data menjadi set pelatihan (80%) dan pengujian (20%)
    train_size = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    # Parameter yang sudah ditentukan
    arima_order = (5, 1, 5)
    seasonal_order = (1, 0, 1, 7)

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Menyimpan hasil model
    models = {}
    total_models = sum([use_arima, use_sarimax])
    progress_step = 1.0 / total_models if total_models > 0 else 1.0

    # ARIMA
    if use_arima:
        status_text.text(f"Melatih model ARIMA{arima_order}...")
        try:
            arima_model = ARIMA(train_df[selected_throughput], order=arima_order).fit()
            ar_train_pred = arima_model.fittedvalues
            ar_test_pred = arima_model.forecast(steps=len(test_df))
            
            mse_arima = mean_squared_error(test_df[selected_throughput], ar_test_pred)
            mape_arima = mean_absolute_percentage_error(test_df[selected_throughput], ar_test_pred)
            
            models['ARIMA'] = {
                'train_pred': ar_train_pred,
                'test_pred': ar_test_pred,
                'mse': mse_arima,
                'mape': mape_arima,
                'color': 'green'
            }
        except Exception as e:
            st.error(f"Kesalahan saat melatih model ARIMA: {str(e)}")
        
        progress_bar.progress(progress_step)
    
    # SARIMAX
    if use_sarimax:
        status_text.text(f"Melatih model SARIMAX{arima_order}{seasonal_order}...")
        try:
            sarimax_model = SARIMAX(
                train_df[selected_throughput], 
                order=arima_order, 
                seasonal_order=seasonal_order
            ).fit(disp=False)
            
            sarimax_train_pred = sarimax_model.fittedvalues
            sarimax_test_pred = sarimax_model.forecast(steps=len(test_df))
            
            mse_sarimax = mean_squared_error(test_df[selected_throughput], sarimax_test_pred)
            mape_sarimax = mean_absolute_percentage_error(test_df[selected_throughput], sarimax_test_pred)
            
            models['SARIMAX'] = {
                'train_pred': sarimax_train_pred,
                'test_pred': sarimax_test_pred,
                'mse': mse_sarimax,
                'mape': mape_sarimax,
                'color': 'purple'
            }
        except Exception as e:
            st.error(f"Kesalahan saat melatih model SARIMAX: {str(e)}")
        
        progress_bar.progress(1.0)
    
    # Menampilkan hasil jika ada model yang dilatih
    if models:
        status_text.text("Semua model telah dilatih! Menampilkan hasil...")

        # Visualisasi hasil
        plt.figure(figsize=(12, 6))
        train_df[selected_throughput].plot(style='--', color='gray', legend=True, label='Data Latihan')
        test_df[selected_throughput].plot(style='--', color='red', legend=True, label='Data Uji')
        
        for model_name, model_data in models.items():
            plt.plot(
                model_data['train_pred'].index, 
                model_data['train_pred'], 
                color=model_data['color'], 
                label=f'Prediksi Latihan {model_name}'
            )
            plt.plot(
                model_data['test_pred'].index, 
                model_data['test_pred'], 
                color=model_data['color'], 
                linestyle='-', 
                label=f'Prediksi Uji {model_name}'
            )

        plt.title(f'Perbandingan Model Peramalan - {selected_throughput}')
        plt.xlabel('Tanggal')
        plt.ylabel(selected_throughput)
        plt.legend()
        st.pyplot(plt)

        # Tabel perbandingan model
        st.subheader("Perbandingan Kinerja Model")
        model_comparison = {
            'Model': [],
            'Mean Squared Error (MSE)': [],
            'Mean Absolute Percentage Error (MAPE)': []
        }
        
        for model_name, model_data in models.items():
            model_comparison['Model'].append(model_name)
            model_comparison['Mean Squared Error (MSE)'].append(f"{model_data['mse']:.4f}")
            model_comparison['Mean Absolute Percentage Error (MAPE)'].append(f"{model_data['mape']:.4f}")
        
        # menampilkan dataframe
        comparison_df = pd.DataFrame(model_comparison)
        st.dataframe(comparison_df)
        
        # menampilkan konfigurasi model
        st.subheader("Model Configurations")
        model_configs = {
            'Model': [],
            'Configuration': []
        }
        
        if 'ARIMA' in models:
            model_configs['Model'].append('ARIMA')
            model_configs['Configuration'].append(f"Order: (p={arima_order[0]}, d={arima_order[1]}, q={arima_order[2]})")
        
        if 'SARIMAX' in models:
            model_configs['Model'].append('SARIMAX')
            model_configs['Configuration'].append(
                f"Order: (p={arima_order[0]}, d={arima_order[1]}, q={arima_order[2]}), " +
                f"Seasonal Order: (P={seasonal_order[0]}, D={seasonal_order[1]}, Q={seasonal_order[2]}, s={seasonal_order[3]})"
            )
        
        st.dataframe(pd.DataFrame(model_configs))
        
        # infokan best model
        if comparison_df.shape[0] > 0:
            best_model_mse = comparison_df.iloc[pd.to_numeric(comparison_df['Mean Squared Error (MSE)']).idxmin()]['Model']
            best_model_mape = comparison_df.iloc[pd.to_numeric(comparison_df['Mean Absolute Percentage Error (MAPE)']).idxmin()]['Model']
            
            st.info(f"Best model based on MSE: {best_model_mse}")
            st.info(f"Best model based on MAPE: {best_model_mape}")
