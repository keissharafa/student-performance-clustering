import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

def run_kmeans(
    input_path='data/output/cleaned_data.csv',
    assets_dir='assets/'
):
    print("=== STEP 2: K-MEANS CLUSTERING (JALUR KEI) ===")
    
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Read cleaned data
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} tidak ditemukan! Jalankan preprocessing.py terlebih dahulu.")
        
    df = pd.read_csv(input_path)
    print(f"Data berhasil dimuat. Bentuk data: {df.shape}")

    # 2. Cek Outlier Visualisasi (Boxplot)
    print("STATUS: Menyimpan grafik Boxplot ke assets/...")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df[['Hours_Studied', 'Attendance', 'Exam_Score']])
    plt.title('Cek Outlier Fitur Numerik (Jam Belajar, Kehadiran, Nilai Ujian)')
    plt.savefig(os.path.join(assets_dir, '1_cek_outlier.png'))
    plt.close()

    # 3. Distribusi Data
    print("STATUS: Menyimpan grafik Distribusi Data ke assets/...")
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(15, 6))
    for i, col in enumerate(['Hours_Studied', 'Attendance', 'Exam_Score'], 1):
        plt.subplot(1, 3, i)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[col], kde=True, stat="density", kde_kws=dict(cut=3), bins=20)
        plt.title(f'Distplot of {col}')
    plt.savefig(os.path.join(assets_dir, '2_distribusi_data.png'))
    plt.close()

    # 4. Regplot (Korelasi)
    print("STATUS: Menyimpan grafik Regplot Korelasi ke assets/...")
    plt.figure(figsize=(15, 15))
    n = 0
    for x in ['Hours_Studied', 'Attendance', 'Exam_Score']:
        for y in ['Hours_Studied', 'Attendance', 'Exam_Score']:
            n += 1
            plt.subplot(3, 3, n)
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            sns.regplot(x=x, y=y, data=df)
            plt.ylabel(y.split()[0] + ' ' + y.split()[1] if len(y.split()) > 1 else y)
    plt.savefig(os.path.join(assets_dir, '3_korelasi_regplot.png'))
    plt.close()

    # 5. Scatter Plot berdasarkan Gender
    print("STATUS: Menyimpan Scatter Plot Gender ke assets/...")
    plt.figure(figsize=(15, 8))
    for gender in ['Male', 'Female']:
        subset = df[df['Gender'] == gender]
        plt.scatter(x=subset['Attendance'], y=subset['Exam_Score'], s=200, alpha=0.5, label=gender)
    plt.xlabel('Attendance')
    plt.ylabel('Exam Score')
    plt.title('Attendance vs Exam Score by Gender')
    plt.legend()
    plt.savefig(os.path.join(assets_dir, '4_scatter_gender.png'))
    plt.close()

    # ==========================================
    # PEMODELAN K-MEANS
    # ==========================================
    X1 = df[['Attendance', 'Exam_Score']].values

    # 6. Elbow Method
    print("STATUS: Mencari K optimal via Elbow Method...")
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=111)
        kmeans.fit(X1)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(15, 6))
    plt.plot(range(1, 11), inertia, 'o')
    plt.plot(range(1, 11), inertia, '-', alpha=0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method (Cari Patahan Siku)')
    plt.savefig(os.path.join(assets_dir, '5_grafik_elbow_kmeans.png'))
    plt.close()

    # 7. Training K-Means Utama
    jumlah_k_optimal = 4 
    print(f"STATUS: Membangun K-Means dengan K={jumlah_k_optimal}...")
    algorithm = KMeans(
        n_clusters=jumlah_k_optimal, 
        init='k-means++', 
        n_init=10, 
        max_iter=300, 
        tol=0.0001, 
        random_state=111, 
        algorithm='elkan'
    )
    algorithm.fit(X1)
    labels2 = algorithm.labels_
    centroids2 = algorithm.cluster_centers_

    # 8. Visualisasi Klaster
    step = 0.5
    x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
    y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(15, 7))
    Z1 = Z1.reshape(xx.shape)
    plt.imshow(
        Z1, interpolation='nearest', 
        extent=(xx.min(), xx.max(), yy.min(), yy.max()), 
        cmap=plt.cm.Pastel2, aspect='auto', origin='lower'
    )
    plt.scatter(x=X1[:, 0], y=X1[:, 1], c=labels2, s=100)
    plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=300, c='red', alpha=0.8, marker='X')
    plt.ylabel('Exam Score')
    plt.xlabel('Attendance')
    plt.title(f'Visualisasi Cluster K-Means (K={jumlah_k_optimal})')
    plt.savefig(os.path.join(assets_dir, '6_hasil_klaster_kmeans.png'))
    plt.close()

    # 9. Evaluasi Silhouette Score
    score2 = silhouette_score(X1, labels2)
    print("==========================================")
    print(f"Hasil Evaluasi Silhouette Score (K-Means): {score2:.4f}")
    print("==========================================\n")
    
    return score2

if __name__ == '__main__':
    run_kmeans()