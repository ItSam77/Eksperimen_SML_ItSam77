import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def preprocess_data(file_path):
    """
    Memuat dan melakukan preprocessing data dari file CSV berdasarkan
    langkah-langkah di notebook Template_Eksperimen_MSML.ipynb.

    Args:
        file_path (str): Path menuju file CSV input ('loan_approval_dataset.csv').

    Returns:
        tuple: Berisi:
            - pd.DataFrame: DataFrame yang sudah diproses dan siap dilatih.
            - dict: Mapping (kamus) dari kolom kategorikal dan nilai-nilai
                    hasil encodingnya.
            Mengembalikan (None, None) jika file tidak ditemukan.
    """
    print(f"Memulai preprocessing untuk file: {file_path}")

    # 1. Memuat Dataset
    try:
        # Menghapus spasi di awal/akhir nama kolom saat loading
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        print("Dataset berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di '{file_path}'")
        return None, None
    except Exception as e:
        print(f"Error saat memuat dataset: {e}")
        return None, None

    # 2. Menghapus Kolom yang Tidak Diperlukan ('loan_id')
    if 'loan_id' in df.columns:
        df.drop(columns=['loan_id'], inplace=True)
        print("Kolom 'loan_id' berhasil dihapus.")
    else:
        print("Kolom 'loan_id' tidak ditemukan, langkah ini dilewati.")

    # 3. Menangani Duplikat
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_after_duplicates = len(df)
    print(f"Menangani duplikat: {initial_rows - rows_after_duplicates} baris duplikat dihapus.")

    # 4. Identifikasi Kolom Kategorikal (berdasarkan notebook)
    categorical_cols = ['education', 'self_employed']
    target_col = 'loan_status'

    # Filter kolom yang benar-benar ada di DataFrame
    actual_categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    if target_col not in df.columns:
        print(f"Error: Kolom target '{target_col}' tidak ditemukan.")
        return None, None

    print(f"Kolom kategorikal yang akan di-encode: {actual_categorical_cols}")
    print(f"Kolom target: {target_col}")

    # 5. Label Encoding
    encoded_columns_mapping = {}
    
    # Encoding Fitur Kategorikal
    for col in actual_categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoded_columns_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Encoding Kolom Target
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    encoded_columns_mapping[target_col] = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))

    print("Label Encoding selesai.")
    print("\nHasil Mapping Encoding:")
    for col, mapping in encoded_columns_mapping.items():
        print(f"- {col}: {mapping}")

    print("\nPreprocessing selesai. Data siap untuk dilatih.")
    return df, encoded_columns_mapping

# --- Contoh Penggunaan ---
if __name__ == "__main__":
    # Pastikan file 'loan_approval_dataset.csv' ada di direktori yang sama
    # atau ganti dengan path yang benar.
    input_csv_path = 'loan_approval_dataset.csv'

    processed_df, mappings = preprocess_data(input_csv_path)

    if processed_df is not None:
        print("\n--- 5 Baris Pertama Data Hasil Preprocessing ---")
        print(processed_df.head())
        print("\n--- Info Data Hasil Preprocessing ---")
        processed_df.info()