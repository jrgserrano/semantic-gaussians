import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

def generate_pairs_from_odom(csv_path, output_pairs_path, num_neighbors=15, max_distance=2.0):
    df = pd.read_csv(csv_path)
    
    # Extraer posiciones x, y, z
    coords = df[['x', 'y', 'z']].values
    names = df['label'].values
    
    print(f"[INFO] Buscando {num_neighbors} vecinos para cada una de las {len(df)} imágenes...")
    
    # Buscar vecinos más cercanos en el espacio 3D
    nn = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm='kd_tree')
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    
    pairs = set()
    for i in range(len(df)):
        for j_idx, dist in zip(indices[i], distances[i]):
            if i == j_idx: continue
            if dist > max_distance: continue
            
            # Ordenar nombres para evitar duplicados (A,B) y (B,A)
            pair = tuple(sorted([names[i], names[j_idx]]))
            pairs.add(pair)
            
    # También añadir parejas consecutivas (temporalmente cerca)
    for i in range(len(df) - 1):
        pair = tuple(sorted([names[i], names[i+1]]))
        pairs.add(pair)

    print(f"[OK] Generadas {len(pairs)} parejas únicas para matching.")
    
    with open(output_pairs_path, "w") as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    generate_pairs_from_odom(args.csv, args.out)
