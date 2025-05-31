import pandas as pd
import warnings
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def preprocess_data(file_path):
    """
    Memuat dan melakukan preprocessing data dari file CSV berdasarkan
    langkah-langkah di notebook Template_Eksperimen_MSML.ipynb.

    Args:
        file_path (str): Path menuju file CSV input ('loan_approval_dataset.csv').

    Returns:
        tuple: 
            - pd.DataFrame: DataFrame yang sudah diproses dan siap dilatih.
            - dict: Mapping dari hasil encoding kolom kategorikal dan target.
    """
    print(f"Memulai preprocessing untuk file: {file_path}")

    # 1. Memuat Dataset
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()  # Menghapus spasi di nama kolom
        print("✅ Dataset berhasil dimuat.")
    except FileNotFoundError:
        print(f"❌ File tidak ditemukan di '{file_path}'")
        return None, None
    except Exception as e:
        print(f"❌ Error saat memuat dataset: {e}")
        return None, None

    # 2. Hapus Kolom Tidak Diperlukan
    if 'loan_id' in df.columns:
        df.drop(columns=['loan_id'], inplace=True)
        print("✅ Kolom 'loan_id' berhasil dihapus.")
    else:
        print("ℹ️ Kolom 'loan_id' tidak ditemukan, dilewati.")

    # 3. Hapus Duplikat
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"✅ Duplikat dihapus: {before - after} baris.")

    # 4. Encoding Kolom Kategorikal & Target
    categorical_cols = ['education', 'self_employed']
    target_col = 'loan_status'

    actual_categorical_cols = [col for col in categorical_cols if col in df.columns]

    if target_col not in df.columns:
        print(f"❌ Kolom target '{target_col}' tidak ditemukan.")
        return None, None

    print(f"📦 Kolom kategorikal: {actual_categorical_cols}")
    print(f"🎯 Kolom target: {target_col}")

    encoded_columns_mapping = {}

    for col in actual_categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoded_columns_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    encoded_columns_mapping[target_col] = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))

    print("✅ Label encoding selesai.")

    # 5. Standardisasi Kolom Numerik
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != target_col and not col.startswith(('Department_', 'Gender_'))]

    if numerical_cols:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print(f"✅ Standardisasi selesai untuk kolom: {numerical_cols}")
    else:
        print("ℹ️ Tidak ada kolom numerik untuk distandarisasi.")

    # 6. Simpan hasil preprocessing
    output_dir = "preprocessing/loan_approval_preprocessing"
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "loan_approval_cleaned.csv")
    df.to_csv(output_file_path, index=False)
    print(f"✅ Data hasil preprocessing disimpan di: {output_file_path}")

    return df, encoded_columns_mapping

# --- Contoh Penggunaan ---
if __name__ == "__main__":
    input_csv_path = "../data.csv"
    processed_df, mappings = preprocess_data(input_csv_path)

    if processed_df is not None:
        print("\n--- 5 Baris Pertama Data Hasil Preprocessing ---")
        print(processed_df.head())
        print("\n--- Info Data Hasil Preprocessing ---")
        processed_df.info()
