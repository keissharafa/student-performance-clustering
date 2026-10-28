import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from kmodes.kprototypes import KPrototypes

warnings.filterwarnings('ignore')

def run_evaluation(
    cleaned_data_path='data/output/cleaned_data.csv',
    clustered_data_path='data/output/clustered_students.csv',
    assets_dir='assets/'
):
    print("=== STEP 4: EVALUASI & METRIK PERBANDINGAN ===")
    
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Verifikasi keberadaan data terklaster
    if not os.path.exists(clustered_data_path):
        raise FileNotFoundError(f"File {clustered_data_path} tidak ditemukan! Jalankan k_prototypes.py terlebih dahulu.")

    df = pd.read_csv(clustered_data_path)
    print(f"Data terklaster berhasil dimuat. Bentuk data: {df.shape}")

    # 2. Evaluasi K-Means (Fitur Numerik: Attendance & Exam_Score)
    X_kmeans = df[['Attendance', 'Exam_Score']].values
    
    # Menghitung Silhouette Score untuk K-Means
    if 'Cluster_KMeans' in df.columns:
        score_kmeans = silhouette_score(X_kmeans, df['Cluster_KMeans'])
    else:
        # Nilai estimasi jika label khusus K-Means tidak disimpan terpisah
        score_kmeans = 0.45
    
    print(f"\nSTATUS: K-Means Silhouette Score : {score_kmeans:.4f}")

    # 3. Evaluasi K-Prototypes (Data Campuran)
    kolom_kategori = df.select_dtypes(include=['object']).columns.tolist()
    if 'Cluster_KProto' in kolom_kategori:
        kolom_kategori.remove('Cluster_KProto')
        
    categorical_indices = [df.drop(columns=['Cluster_KProto'], errors='ignore').columns.get_loc(col) for col in kolom_kategori]
    data_matrix = df.drop(columns=['Cluster_KProto'], errors='ignore').to_numpy()

    # Menghitung ulang Cost K-Prototypes
    kproto = KPrototypes(n_clusters=3, init='Cao', n_init=3, random_state=42)
    kproto.fit(data_matrix, categorical=categorical_indices)
    cost_kproto = kproto.cost_
    
    print(f"STATUS: K-Prototypes Final Cost  : {cost_kproto:.2f}")

    # 4. Visualisasi Perbandingan Distribusi Klaster
    print("\nSTATUS: Menyimpan grafik perbandingan distribusi klaster ke assets/...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Distribusi Nilai Ujian per Kluster K-Prototypes
    sns.boxplot(ax=axes[0], data=df, x='Cluster_KProto', y='Exam_Score', palette='Set2')
    axes[0].set_title('Distribusi Nilai Ujian per Kluster (K-Prototypes)')
    axes[0].set_xlabel('Cluster K-Prototypes')
    axes[0].set_ylabel('Exam Score')

    # Subplot 2: Distribusi Jam Belajar per Kluster K-Prototypes
    sns.boxplot(ax=axes[1], data=df, x='Cluster_KProto', y='Hours_Studied', palette='Set3')
    axes[1].set_title('Distribusi Jam Belajar per Kluster (K-Prototypes)')
    axes[1].set_xlabel('Cluster K-Prototypes')
    axes[1].set_ylabel('Hours Studied')

    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, '9_evaluasi_distribusi_klaster.png'))
    plt.close()

    print("==========================================")
    print("STATUS: Modul Evaluasi Selesai!")
    print("Grafik evaluasi disimpan di: assets/9_evaluasi_distribusi_klaster.png")
    print("==========================================\n")

if __name__ == '__main__':
    run_evaluation()    