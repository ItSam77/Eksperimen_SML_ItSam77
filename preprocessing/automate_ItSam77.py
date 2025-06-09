import pandas as pd
import numpy as np
import warnings
import os
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def preprocess_data(file_path):
    """
    Memuat dan melakukan preprocessing data dari file CSV berdasarkan
    langkah-langkah di notebook Template_Eksperimen_MSML.ipynb.

    Args:
        file_path (str): Path menuju file CSV input ('data.csv').

    Returns:
        tuple: 
            - dict: Dictionary berisi data hasil preprocessing dengan kunci:
                - 'X_train': Features untuk training
                - 'X_test': Features untuk testing
                - 'y_train': Target untuk training
                - 'y_test': Target untuk testing
                - 'preprocessed_df': DataFrame yang sudah diproses lengkap
            - dict: Mapping dari hasil encoding kolom kategorikal dan target.
    """
    print(f"🚀 Memulai preprocessing untuk file: {file_path}")
    print("=" * 60)

    # 1. Memuat Dataset
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # Menghapus spasi di nama kolom
        print("✅ Dataset berhasil dimuat.")
        print(f"📊 Bentuk dataset: {df.shape}")
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan di '{file_path}'")
        return None, None
    except Exception as e:
        print(f"❌ Error saat memuat dataset: {e}")
        return None, None

    print("\n" + "=" * 60)
    print("📋 EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 2. EDA - Informasi dasar dataset
    print(f"📏 Shape dataset: {df.shape}")
    print(f"🔍 Info dataset:")
    df.info()
    
    print(f"\n📊 5 baris pertama:")
    print(df.head())
    
    print(f"\n📊 5 baris terakhir:")
    print(df.tail())
    
    print(f"\n📈 Deskripsi statistik:")
    print(df.describe())
    
    # Cek missing values
    print(f"\n🔍 Missing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        print("✅ Tidak ada missing values ditemukan.")
    
    # Cek data types
    print(f"\n🏷️ Tipe data:")
    print(df.dtypes)
    
    # Cek unique values untuk setiap kolom
    print(f"\n🔢 Unique values di setiap kolom:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    # Cek duplikat
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Duplicate rows: {duplicates}")

    print("\n" + "=" * 60)
    print("🧹 DATA CLEANING & PREPROCESSING")
    print("=" * 60)

    # 3. Hapus Kolom Tidak Diperlukan (untuk employee attrition dataset)
    columns_to_drop = ['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)
        print(f"✅ Kolom tidak diperlukan berhasil dihapus: {existing_columns_to_drop}")
    else:
        print("ℹ️ Kolom yang akan dihapus tidak ditemukan, dilewati.")

    # 4. Hapus Duplikat
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"✅ Duplikat dihapus: {before - after} baris.")
    else:
        print("✅ Tidak ada duplikat ditemukan.")

    # 5. Handling Missing Values (jika ada)
    if df.isnull().sum().sum() > 0:
        print("🔧 Menangani missing values...")
        # Untuk kolom numerik, isi dengan median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  - {col}: diisi dengan median")
        
        # Untuk kolom kategorikal, isi dengan modus
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  - {col}: diisi dengan modus")
    else:
        print("✅ Tidak ada missing values yang perlu ditangani.")

    # 6. Identifikasi kolom kategorikal dan numerik
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 
                       'JobRole', 'MaritalStatus', 'OverTime']
    target_col = 'Attrition'

    actual_categorical_cols = [col for col in categorical_cols if col in df.columns]

    if target_col not in df.columns:
        print(f"❌ Kolom target '{target_col}' tidak ditemukan.")
        return None, None

    print(f"📦 Kolom kategorikal ditemukan: {actual_categorical_cols}")
    print(f"🎯 Kolom target: {target_col}")

    # 7. Label Encoding untuk kolom kategorikal & target
    encoded_columns_mapping = {}

    for col in actual_categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoded_columns_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"  - {col}: {len(le.classes_)} kategori diubah ke numerik")

    # Label encoding untuk target
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    encoded_columns_mapping[target_col] = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
    print(f"✅ Label encoding selesai untuk {len(actual_categorical_cols)} kolom kategorikal + target.")

    print("\n" + "=" * 60)
    print("🧹 REMOVING REDUNDANT FEATURES")
    print("=" * 60)

    # 8. Drop redundant features based on analysis
    redundant_features = [
        'EmployeeNumber',      # ID column, not useful for prediction
        'MonthlyIncome',       # Redundant with JobLevel
        'PercentSalaryHike',   # Redundant with PerformanceRating
        'YearsInCurrentRole',  # Redundant with YearsAtCompany
        'YearsWithCurrManager',# Redundant with YearsAtCompany
        'Department'           # Redundant with JobRole
    ]

    # Remove redundant features from the dataset
    existing_redundant = [col for col in redundant_features if col in df.columns]
    if existing_redundant:
        df = df.drop(columns=existing_redundant, errors='ignore')
        print(f"✅ Redundant features berhasil dihapus: {existing_redundant}")
        print(f"📊 Shape dataset setelah menghapus redundant features: {df.shape}")
    else:
        print("ℹ️ Tidak ada redundant features yang perlu dihapus.")

    print("\n" + "=" * 60)
    print("🎯 FEATURE SELECTION")
    print("=" * 60)

    # 9. Pisahkan features dan target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"📊 Total features sebelum seleksi: {X.shape[1]}")
    print(f"🎯 Target shape: {y.shape}")
    print(f"📈 Target distribution:")
    print(y.value_counts())

    # 10. Feature Selection menggunakan SelectKBest
    print(f"\n🔍 Melakukan feature selection dengan SelectKBest...")
    print(f"📊 Memilih 10 features terbaik dari {X.shape[1]} features")
    
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    
    # Dapatkan nama-nama features yang terpilih
    selected_features = X.columns[selector.get_support()]
    feature_scores = selector.scores_[selector.get_support()]
    
    print(f"✅ {len(selected_features)} features terbaik berhasil dipilih!")
    print(f"\n📋 Top 10 Features yang dipilih:")
    for i, (feature, score) in enumerate(zip(selected_features, feature_scores), 1):
        print(f"  {i:2d}. {feature:<25} (Score: {score:.3f})")
    
    # Buat DataFrame baru dengan features yang dipilih
    df_selected = pd.DataFrame(X_selected, columns=selected_features)
    df_selected[target_col] = y.values
    
    print(f"\n📏 Shape dataset setelah feature selection: {df_selected.shape}")

    print("\n" + "=" * 60)
    print("💾 MENYIMPAN HASIL")
    print("=" * 60)

   
    
    # Simpan dataset dengan features terpilih
    output_file_path = os.path.join("processed_data.csv")
    df_selected.to_csv(output_file_path, index=False)
    print(f"✅ Dataset yang sudah diproses disimpan di: {output_file_path}")

    print("\n" + "=" * 60)
    print("🎉 PREPROCESSING SELESAI!")
    print("=" * 60)
    print(f"✅ Data siap dengan {len(selected_features)} features terpilih")
    print(f"📊 Shape akhir: {df_selected.shape}")
    
    return df_selected, encoded_columns_mapping

# --- Contoh Penggunaan ---
if __name__ == "__main__":
    input_csv_path = "../data.csv"
    processed_data, mappings = preprocess_data(input_csv_path)

    if processed_data is not None:
        print("\n" + "=" * 60)
        print("📋 PREVIEW HASIL PREPROCESSING")
        print("=" * 60)
        
        print("\n--- 5 Baris Pertama Dataset ---")
        print(processed_data.head())
        
        print("\n--- Info Dataset Final ---")
        processed_data.info()
        
        print("\n--- Mapping Encoding ---")
        for col, mapping in mappings.items():
            print(f"{col}: {mapping}")
