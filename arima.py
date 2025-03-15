import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
from itertools import product

# Judul aplikasi Streamlit
st.title('Prediksi QoS dengan ARIMA (75% Pelatihan, 25% Pengujian)')

# Pengunggah file CSV
uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])

if uploaded_file is not None:
    # Muat dataset
    df = pd.read_csv(uploaded_file)

    # Ubah kolom 'Tanggal' menjadi datetime dan tetapkan sebagai indeks
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.set_index('Tanggal', inplace=True)

    # Tampilkan data
    st.write("Pratinjau Data:")
    st.dataframe(df)

    # Bagi data menjadi set pelatihan dan pengujian (80% pelatihan, 20% pengujian)
    train_size = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    # Fungsi untuk mencari parameter ARIMA terbaik
    def optimalkan_arima(deret, max_p=3, max_d=2, max_q=3):
        best_aic = np.inf
        best_order = None

        # Ulangi untuk semua kombinasi (p, d, q)
        for p, d, q in product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
            try:
                model = ARIMA(deret, order=(p, d, q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
            except:
                continue
        return best_order

    # Peramalan untuk setiap kolom
    langkah_peramalan = 4  # Langkah peramalan tetap untuk kesederhanaan
    hasil = []

    for kolom in df.columns:
        st.subheader(f"Peramalan untuk {kolom}")

        # Cari parameter ARIMA terbaik
        urutan_terbaik = optimalkan_arima(train_df[kolom])
        st.write(f"Parameter terbaik untuk {kolom}: p={urutan_terbaik[0]}, d={urutan_terbaik[1]}, q={urutan_terbaik[2]}")

        # Sesuaikan model ARIMA
        model = ARIMA(train_df[kolom], order=urutan_terbaik)
        model_disesuaikan = model.fit()

        # Ramalkan nilai masa depan
        peramalan = model_disesuaikan.forecast(steps=langkah_peramalan)
        indeks_peramalan = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=langkah_peramalan)

        # Simpan hasil
        df_peramalan = pd.DataFrame({'Tanggal': indeks_peramalan, kolom: peramalan.values})
        hasil.append(df_peramalan)

        # Pembuatan plot dari prediksi ARIMA
        plt.figure(figsize=(10, 4))

        # Plot data pelatihan asli dengan gaya garis putus-putus
        plt.plot(train_df[kolom], label='Data Asli (Pelatihan)', color='blue', linestyle='--')

        # Plot data pengujian dengan warna abu-abu
        plt.plot(test_df[kolom], label='Data Pengujian', color='gray', linestyle='--')

        # Plot fitted values dari model
        plt.plot(model_disesuaikan.fittedvalues.index, model_disesuaikan.fittedvalues,
                 label='Nilai Disesuaikan', color='orange')

        # Plot nilai yang diramalkan
        plt.plot(indeks_peramalan, peramalan.values, label='Peramalan', color='green')

        # Kustomisasi sumbu x dan label
        plt.gca().xaxis.set_major_locator(DayLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()

        plt.title(f'Prediksi ARIMA untuk {kolom}')
        plt.xlabel('Tanggal')
        plt.ylabel(kolom)
        plt.legend()
        st.pyplot(plt)

    # Gabungkan semua peramalan menjadi satu DataFrame
    peramalan_akhir = pd.concat([hasil[0]['Tanggal']] + [df[kolom] for df, kolom in zip(hasil, df.columns)], axis=1)

    # Tampilkan peramalan akhir
    peramalan_akhir.insert(0, 'No', range(1, len(peramalan_akhir) + 1))

    # Tampilkan peramalan akhir
    st.write("Hasil Peramalan:")
    st.dataframe(peramalan_akhir.reset_index(drop=True), hide_index=True)

    # Opsional, simpan ke file CSV
    csv = peramalan_akhir.to_csv(index=False)
    st.download_button("Unduh Hasil Peramalan", data=csv, file_name="forecast_results.csv", mime="text/csv")
