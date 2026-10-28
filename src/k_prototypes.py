import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kprototypes import KPrototypes

warnings.filterwarnings('ignore')

def run_kprototypes(
    input_path='data/output/cleaned_data.csv',
    output_path='data/output/clustered_students.csv',
    assets_dir='assets/'
):
    print("=== STEP 3: K-PROTOTYPES CLUSTERING ===")
    
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Read cleaned data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} tidak ditemukan! Jalankan preprocessing.py terlebih dahulu.")
        
    df = pd.read_csv(input_path)
    print(f"Data berhasil dimuat. Bentuk data: {df.shape}")

    # 2. Identifikasi Kolom & Indeks Kategorikal
    kolom_kategori = df.select_dtypes(include=['object']).columns.tolist()
    categorical_indices = [df.columns.get_loc(col) for col in kolom_kategori]
    
    print(f"Terdeteksi {len(kolom_kategori)} kolom kategorikal.")

    # Konversi dataframe ke array NumPy untuk K-Prototypes
    data_matrix = df.to_numpy()

    # ==========================================
    # PEMODELAN K-PROTOTYPES
    # ==========================================

    # 3. Menentukan Nilai K Optimal (Elbow Method / Cost Analysis)
    print("STATUS: Mencari K optimal via Cost Function (K-Prototypes)...")
    costs = []
    k_range = range(1, 7)
    
    for k in k_range:
        kproto_test = KPrototypes(n_clusters=k, init='Cao', n_init=3, random_state=42)
        kproto_test.fit(data_matrix, categorical=categorical_indices)
        costs.append(kproto_test.cost_)

    # Plot grafik visual Elbow K-Prototypes
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, costs, 'o-r', alpha=0.7)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Cost (Inertia Equivalent)')
    plt.title('Elbow Method K-Prototypes (Cari Patahan Siku)')
    plt.grid(True)
    plt.savefig(os.path.join(assets_dir, '7_grafik_elbow_kprototypes.png'))
    plt.close()

    # 4. Membangun K-Prototypes Utama
    jumlah_k_optimal = 3
    print(f"STATUS: Membangun K-Prototypes dengan K={jumlah_k_optimal}...")
    
    kproto = KPrototypes(n_clusters=jumlah_k_optimal, init='Cao', n_init=5, random_state=42)
    clusters = kproto.fit_predict(data_matrix, categorical=categorical_indices)

    # 5. Menambahkan Label Kluster ke Dataframe & Ekspor Data
    df['Cluster_KProto'] = clusters
    
    print("\n--- Ringkasan Jumlah Siswa Per Kluster ---")
    print(df['Cluster_KProto'].value_counts())

    print("\n--- Pusat Kluster (Centroids) ---")
    print(kproto.cluster_centroids_)

    df.to_csv(output_path, index=False)
    print(f"\nSTATUS: Data siswa dengan label kluster disimpan ke '{output_path}'")

    # 6. Visualisasi Hasil Kluster K-Prototypes
    print("STATUS: Menyimpan grafik Scatter Plot Klaster K-Prototypes ke assets/...")
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df,
        x='Hours_Studied',
        y='Exam_Score',
        hue='Cluster_KProto',
        style='Parental_Involvement',
        palette='Set1',
        s=100,
        alpha=0.8
    )
    plt.title(f'Visualisasi Cluster K-Prototypes (K={jumlah_k_optimal})')
    plt.xlabel('Hours Studied')
    plt.ylabel('Exam Score')
    plt.legend(title='Cluster / Parental Involvement', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '8_hasil_klaster_kprototypes.png'))
    plt.close()
    
    print("==========================================")
    print(f"Hasil Evaluasi Final Cost K-Prototypes: {kproto.cost_:.2f}")
    print("==========================================\n")
    return df

if __name__ == '__main__':
    run_kprototypes()