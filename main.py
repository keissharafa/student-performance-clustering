import os
import sys
import time

# Menambahkan folder src ke sys.path agar modul di dalamnya bisa di-import dengan bersih
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from preprocessing import run_preprocessing
    from k_means import run_kmeans
    from k_prototypes import run_kprototypes
    from evaluation import run_evaluation
except ImportError as e:
    print(f"[ERROR] Gagal meng-import modul dari folder src: {e}")
    sys.exit(1)


def main():
    print("==========================================================")
    print("   STUDENT PERFORMANCE CLUSTERING PIPELINE (COMPLETE)")
    print("==========================================================\n")
    
    start_time = time.time()

    # TAHAP 1: PREPROCESSING DATA
    print(">>> [1/4] Menjalankan Preprocessing Data...")
    run_preprocessing(
        input_path='data/input/StudentPerformanceFactors.csv',
        output_path='data/output/cleaned_data.csv'
    )

    # TAHAP 2: K-MEANS CLUSTERING
    print("\n>>> [2/4] Menjalankan K-Means Clustering (Data Numerik)...")
    run_kmeans(
        input_path='data/output/cleaned_data.csv',
        assets_dir='assets/'
    )

    # TAHAP 3: K-PROTOTYPES CLUSTERING
    print("\n>>> [3/4] Menjalankan K-Prototypes Clustering (Data Campuran)...")
    run_kprototypes(
        input_path='data/output/cleaned_data.csv',
        output_path='data/output/clustered_students.csv',
        assets_dir='assets/'
    )

    # TAHAP 4: EVALUASI & METRIK VISUALISASI AKHIR
    print("\n>>> [4/4] Menjalankan Evaluasi & Visualisasi Akhir...")
    run_evaluation(
        cleaned_data_path='data/output/cleaned_data.csv',
        clustered_data_path='data/output/clustered_students.csv',
        assets_dir='assets/'
    )

    elapsed_time = time.time() - start_time
    
    print("==========================================================")
    print("                  PIPELINE SUKSES DIJALANKAN!             ")
    print("==========================================================")
    print(f"⏱️ Total Waktu Eksekusi: {elapsed_time:.2f} detik")
    print("📁 Output CSV        : data/output/clustered_students.csv")
    print("📊 Output Visual     : assets/ (1_cek_outlier.png s/d 9_evaluasi_distribusi_klaster.png)")
    print("==========================================================\n")


if __name__ == '__main__':
    main()