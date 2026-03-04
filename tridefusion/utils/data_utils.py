import os
import re
import pandas as pd
from pathlib import Path


def convert_csv_to_xlsx(input_folder, output_folder=None):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder) if output_folder else input_folder

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    csv_files = list(input_folder.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            xlsx_path = output_folder / (csv_file.stem + ".xlsx")
            df.to_excel(xlsx_path, index=False)
            print(f"Converted: {csv_file.name} → {xlsx_path.name}")
        except Exception as e:
            print(f"Error converting {csv_file.name}: {e}")

def normalize_noise(name: str) -> str:
    """
    Normalize noise names so that:
      poisson_01_gauss_0.05 → poisson_1_gauss_0.05
      poisson_04_gauss_0.1  → poisson_4_gauss_0.1
    """
    m = re.match(r"poisson_(\d+)_gauss_([0-9.]+)", name)
    if not m:
        return name

    poisson = int(m.group(1))
    gauss = float(m.group(2))
    gauss_str = f"{gauss:g}"
    return f"poisson_{poisson}_gauss_{gauss_str}"


def split_by_noise(csv_path, output_dir="splits"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df["NoiseNormalized"] = df["NoiseFolder"].apply(normalize_noise)
    for noise_name, df_group in df.groupby("NoiseNormalized"):
        filename = f"{noise_name}.csv"
        path = os.path.join(output_dir, filename)
        df_group.to_csv(path, index=False)
        print(f"Saved: {path}  ({len(df_group)} rows)")
    print("\nDone!")


def reorder_methods(
    csv_path,
    output_csv="reordered.csv",
    method_order=None
):
    if method_order is None:
        raise ValueError("You must specify method_order=[...]")
    df = pd.read_csv(csv_path)

    if "Method" not in df.columns:
        raise ValueError("CSV file does not contain 'Method' column")
    existing_methods = df["Method"].unique().tolist()
    ordered_existing = [m for m in method_order if m in existing_methods]
    remaining = sorted([m for m in existing_methods if m not in method_order])
    final_order = ordered_existing + remaining
    df["Method"] = pd.Categorical(df["Method"], categories=final_order, ordered=True)
    df_sorted = df.sort_values("Method")
    df_sorted.to_csv(output_csv, index=False)
    print(f"✅ Reordered CSV saved to {output_csv}")
    print("Final order:")
    for m in final_order:
        print("  ", m)
    return df_sorted