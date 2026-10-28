import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import warnings

# Mengabaikan pesan warning agar terminal tetap rapi
warnings.filterwarnings('ignore')

os.makedirs('output_visualisasi', exist_ok=True)
print("STATUS: Folder 'output_visualisasi' siap.")

# 2. Import data
# Membaca dataset dari dalam folder 'data'
df = pd.read_csv('data/StudentPerformanceFactors.csv')
print("\n--- 5 Baris Pertama Dataset ---")
print(df.head())

# 3. Amati bentuk data
print("\n--- Bentuk Data (Baris, Kolom) ---")
print(df.shape)

# 4. Melihat ringkasan statistik deskriptif dari DataFrame
print("\n--- Statistik Deskriptif ---")
print(df.describe())

# 5. Cek null data
print("\n--- Jumlah Null Data ---")
print(df.isnull().sum())

# 6. Cek outlier (Menggunakan Boxplot untuk kolom numerik utama)
print("\nSTATUS: Membuat grafik Boxplot (Cek Outlier)...")
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Hours_Studied', 'Attendance', 'Exam_Score']])
plt.title('Cek Outlier Fitur Numerik (Jam Belajar, Kehadiran, Nilai Ujian)')
plt.savefig('output_visualisasi/1_cek_outlier.png')
plt.show()

# 7 & 8. Amati bentuk visual masing-masing fitur (Distribusi Data)
print("STATUS: Membuat grafik Distribusi Data...")
plt.style.use('fivethirtyeight')
plt.figure(1, figsize=(15, 6))
n = 0
for x in ['Hours_Studied', 'Attendance', 'Exam_Score']:
    n += 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    sns.histplot(df[x], kde=True, stat="density", kde_kws=dict(cut=3), bins=20)
    plt.title('Distplot of {}'.format(x))
plt.savefig('output_visualisasi/2_distribusi_data.png')
plt.show()

# 9a. Ploting untuk mencari relasi antar fitur numerik (Regplot)
print("STATUS: Membuat grafik Regplot (Korelasi)...")
plt.figure(1, figsize=(15, 15))
n = 0
for x in ['Hours_Studied', 'Attendance', 'Exam_Score']:
    for y in ['Hours_Studied', 'Attendance', 'Exam_Score']:
        n += 1
        plt.subplot(3, 3, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.regplot(x=x, y=y, data=df)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y)
plt.savefig('output_visualisasi/3_korelasi_regplot.png')
plt.show()

# 9b. Melihat sebaran Attendance dan Exam Score berdasarkan Gender (Scatter Plot)
print("STATUS: Membuat grafik Scatter Plot (Berdasarkan Gender)...")
plt.figure(1, figsize=(15, 8))
for gender in ['Male', 'Female']:
    plt.scatter(x='Attendance', y='Exam_Score', 
                data=df[df['Gender'] == gender], s=200, alpha=0.5, label=gender)
plt.xlabel('Attendance')
plt.ylabel('Exam Score')
plt.title('Attendance vs Exam Score by Gender')
plt.legend()
plt.savefig('output_visualisasi/4_scatter_gender.png')
plt.show()

# ==========================================
# PEMODELAN K-MEANS
# ==========================================

# Memilih fitur untuk dicluster (Kehadiran dan Nilai Ujian)
# Dropna digunakan untuk membuang baris kosong agar K-Means tidak error
X1 = df[['Attendance', 'Exam_Score']].dropna().iloc[:, :].values

# 10. Menentukan nilai k yang sesuai dengan Elbow-Method
print("STATUS: Menjalankan algoritma K-Means untuk mencari nilai K optimal...")
inertia = []
for n in range(1, 11):
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, random_state=111))
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)

# 11 & 12. Plot bentuk visual elbow
plt.figure(1, figsize=(15, 6))
plt.plot(range(1, 11), inertia, 'o')
plt.plot(range(1, 11), inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method (Cari Patahan Siku)')
plt.savefig('output_visualisasi/5_grafik_elbow.png')
plt.show()

# 13. Membangun K-Means Utama
# CATATAN: Ubah nilai 4 di bawah ini jika grafik Elbow menunjukkan siku tajam di angka lain!
jumlah_k_optimal = 4 

print(f"STATUS: Membangun K-Means dengan K={jumlah_k_optimal}...")
algorithm = (KMeans(n_clusters=jumlah_k_optimal, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan'))
algorithm.fit(X1)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

# Menyiapkan data untuk bentuk visual cluster
step = 0.5 # Skala diperbesar agar proses render visualisasi tidak terlalu berat
x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
Z1 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

# Melihat bentuk visual cluster K-Means
plt.figure(1, figsize=(15, 7))
plt.clf()
Z1 = Z1.reshape(xx.shape)
plt.imshow(Z1, interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), 
           cmap=plt.cm.Pastel2, aspect='auto', origin='lower')

plt.scatter(x=X1[:, 0], y=X1[:, 1], c=labels2, s=100)
# Menandai titik pusat (Centroid) dengan tanda X merah
plt.scatter(x=centroids2[:, 0], y=centroids2[:, 1], s=300, c='red', alpha=0.8, marker='X')

plt.ylabel('Exam Score')
plt.xlabel('Attendance')
plt.title(f'Visualisasi Cluster K-Means (K={jumlah_k_optimal})')
plt.savefig('output_visualisasi/6_hasil_klaster_kmeans.png')
plt.show()

# 14. Melihat nilai Silhouette Score
score2 = silhouette_score(X1, labels2)
print("\n==========================================")
print(f"Hasil Evaluasi Silhouette Score: {score2}")
print("==========================================\n")