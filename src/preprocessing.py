import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def run_preprocessing(
    input_path='data/input/StudentPerformanceFactors.csv', 
    output_path='data/output/cleaned_data.csv'
):
    print("=== STEP 1: PREPROCESSING DATA STUDENT PERFORMANCE ===")
    
    # Cek keberadaan file input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} tidak ditemukan! Pastikan file ada di folder data/input/")

    df = pd.read_csv(input_path)
    print(f"Total baris awal dataset: {len(df)}\n")

    # --- 1.1 Deteksi & Handling Duplikat ---
    jumlah_duplikat = df.duplicated().sum()
    print(f"Jumlah data duplikat terdeteksi: {jumlah_duplikat}")
    if jumlah_duplikat > 0:
        df = df.drop_duplicates()
        print("STATUS: Data duplikat berhasil dihapus.\n")
    else:
        print("STATUS: Aman dari duplikasi.\n")

    # --- 1.2 Pemisahan Tipe Data ---
    kolom_kategori = df.select_dtypes(include=['object']).columns.tolist()
    kolom_numerik = df.select_dtypes(exclude=['object']).columns.tolist()

    # --- 1.3 Handling Missing Value ---
    print("Missing value SEBELUM imputasi:")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "Tidak ada missing value.")

    # Imputasi Modus untuk Data Kategorikal
    for col in kolom_kategori:
        if df[col].isnull().sum() > 0:
            modus = df[col].mode()[0]
            df[col] = df[col].fillna(modus)

    # Imputasi Median untuk Data Numerik
    for col in kolom_numerik:
        if df[col].isnull().sum() > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    print("\nMissing value SETELAH imputasi:", df.isnull().sum().sum())

    # --- 1.4 Handling Outlier (IQR Capping khusus numerik) ---
    print("\n--- Analisis Outlier (IQR Capping) ---")
    for col in kolom_numerik:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        batas_bawah = Q1 - (1.5 * IQR)
        batas_atas = Q3 + (1.5 * IQR)
        
        outlier_count = ((df[col] < batas_bawah) | (df[col] > batas_atas)).sum()
        if outlier_count > 0:
            print(f"[{col}] Terdeteksi {outlier_count} outlier -> Batas Min & Max disesuaikan.")
            df[col] = np.where(df[col] < batas_bawah, batas_bawah, df[col])
            df[col] = np.where(df[col] > batas_atas, batas_atas, df[col])

    # Ekspor Data Bersih ke data/output/cleaned_data.csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSTATUS: Preprocessing Selesai! Data disimpan ke '{output_path}'.\n")
    return df

if __name__ == '__main__':
    run_preprocessing()