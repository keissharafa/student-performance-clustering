import os
import sys
import time

# Menambahkan folder src ke sys.path agar modul di dalamnya bisa di-import dengan bersih
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from preprocessing import run_preprocessing
    from k_means import run_kmeans
    from k_prototypes import run_kprototypes
except ImportError as e:
    print(f"[ERROR] Gagal meng-import modul dari folder src: {e}")
    print("Pastikan file preprocessing.py, k-means.py, dan k-prototypes.py ada di dalam folder src/")
    sys.exit(1)


def main():
    print("==========================================================")
    print("   STUDENT PERFORMANCE CLUSTERING PIPELINE (K-MEANS & K-PROTO)")
    print("==========================================================\n")
    
    start_time = time.time()

    # -------------------------------------------------------------------------
    # TAHAP 1: PREPROCESSING DATA
    # -------------------------------------------------------------------------
    print(">>> [1/3] Menjalankan Preprocessing Data...")
    try:
        run_preprocessing(
            input_path='data/input/StudentPerformanceFactors.csv',
            output_path='data/output/cleaned_data.csv'
        )
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat Preprocessing: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # TAHAP 2: K-MEANS CLUSTERING
    # -------------------------------------------------------------------------
    print("\n>>> [2/3] Menjalankan K-Means Clustering (Data Numerik)...")
    try:
        score_kmeans = run_kmeans(
            input_path='data/output/cleaned_data.csv',
            assets_dir='assets/'
        )
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat K-Means: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # TAHAP 3: K-PROTOTYPES CLUSTERING
    # -------------------------------------------------------------------------
    print("\n>>> [3/3] Menjalankan K-Prototypes Clustering (Data Campuran)...")
    try:
        df_clustered = run_kprototypes(
            input_path='data/output/cleaned_data.csv',
            output_path='data/output/clustered_students.csv',
            assets_dir='assets/'
        )
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat K-Prototypes: {e}")
        sys.exit(1)

    elapsed_time = time.time() - start_time
    
    print("\n==========================================================")
    print("                  PIPELINE SUKSES DIJALANKAN!             ")
    print("==========================================================")
    print(f"⏱️ Total Waktu Eksekusi: {elapsed_time:.2f} detik")
    print("📁 Data Hasil Olahan : data/output/clustered_students.csv")
    print("📊 Hasil Visualisasi  : assets/ (1_cek_outlier.png s/d 8_hasil_klaster_kprototypes.png)")
    print("==========================================================\n")


if __name__ == '__main__':
    main()