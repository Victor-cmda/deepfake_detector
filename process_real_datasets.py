"""
Script para processar datasets reais e criar índices para treinamento.
Lê os CSVs do FaceForensics++ e List_of_testing_videos do Celeb-DF.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("PROCESSAMENTO DE DATASETS REAIS")
print("=" * 80)
print()

# Configurações
FF_BASE = "E:/deepfake_detector/data/FaceForensics++/FaceForensics++_C23"
CELEB_BASE = "E:/Celeb"
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# FACEFORENSICS++
# ============================================================================

print("📊 Processando FaceForensics++...")
print("-" * 80)

# CSVs disponíveis
ff_csvs = {
    'original': f'{FF_BASE}/csv/original.csv',
    'Deepfakes': f'{FF_BASE}/csv/Deepfakes.csv',
    'Face2Face': f'{FF_BASE}/csv/Face2Face.csv',
    'FaceSwap': f'{FF_BASE}/csv/FaceSwap.csv',
    'NeuralTextures': f'{FF_BASE}/csv/NeuralTextures.csv',
    'FaceShifter': f'{FF_BASE}/csv/FaceShifter.csv',
    'DeepFakeDetection': f'{FF_BASE}/csv/DeepFakeDetection.csv'
}

# Processar cada CSV
all_ff_videos = []

for category, csv_path in ff_csvs.items():
    if not os.path.exists(csv_path):
        print(f"⚠️  {category}: CSV não encontrado")
        continue
    
    df = pd.read_csv(csv_path)
    
    # Adicionar categoria
    df['category'] = category
    df['dataset'] = 'FaceForensics++'
    
    # Ajustar path para ser absoluto
    df['video_path'] = df['File Path'].apply(lambda x: f"{FF_BASE}/{x}")
    
    # Renomear colunas
    df = df.rename(columns={
        'Label': 'label',
        'Frame Count': 'num_frames',
        'File Size(MB)': 'size_mb'
    })
    
    # Verificar quantos vídeos existem
    existing = df[df['video_path'].apply(os.path.exists)]
    
    print(f"✓ {category:20s}: {len(df):4d} vídeos no CSV, {len(existing):4d} existem")
    
    all_ff_videos.append(existing)

# Combinar todos
df_ff = pd.concat(all_ff_videos, ignore_index=True)

print(f"\n📦 Total FaceForensics++: {len(df_ff)} vídeos")
print(f"   - REAL: {len(df_ff[df_ff['label'] == 'REAL'])}")
print(f"   - FAKE: {len(df_ff[df_ff['label'] == 'FAKE'])}")
print()

# ============================================================================
# CELEB-DF
# ============================================================================

print("📊 Processando Celeb-DF...")
print("-" * 80)

all_celeb_videos = []

# Processar Celeb-real
celeb_real_dir = f"{CELEB_BASE}/Celeb-real"
if os.path.exists(celeb_real_dir):
    videos = [f for f in os.listdir(celeb_real_dir) if f.endswith('.mp4')]
    for video in videos:
        all_celeb_videos.append({
            'video_path': f"{celeb_real_dir}/{video}",
            'label': 'REAL',
            'category': 'Celeb-real',
            'dataset': 'Celeb-DF',
            'video_name': video
        })
    print(f"✓ Celeb-real: {len(videos)} vídeos")

# Processar Celeb-synthesis (FAKE)
celeb_fake_dir = f"{CELEB_BASE}/Celeb-synthesis"
if os.path.exists(celeb_fake_dir):
    videos = [f for f in os.listdir(celeb_fake_dir) if f.endswith('.mp4')]
    for video in videos:
        all_celeb_videos.append({
            'video_path': f"{celeb_fake_dir}/{video}",
            'label': 'FAKE',
            'category': 'Celeb-synthesis',
            'dataset': 'Celeb-DF',
            'video_name': video
        })
    print(f"✓ Celeb-synthesis: {len(videos)} vídeos")

# Processar YouTube-real
youtube_real_dir = f"{CELEB_BASE}/YouTube-real"
if os.path.exists(youtube_real_dir):
    videos = [f for f in os.listdir(youtube_real_dir) if f.endswith('.mp4')]
    for video in videos:
        all_celeb_videos.append({
            'video_path': f"{youtube_real_dir}/{video}",
            'label': 'REAL',
            'category': 'YouTube-real',
            'dataset': 'Celeb-DF',
            'video_name': video
        })
    print(f"✓ YouTube-real: {len(videos)} vídeos")

df_celeb = pd.DataFrame(all_celeb_videos)

if len(df_celeb) > 0:
    print(f"\n📦 Total Celeb-DF: {len(df_celeb)} vídeos")
    print(f"   - REAL: {len(df_celeb[df_celeb['label'] == 'REAL'])}")
    print(f"   - FAKE: {len(df_celeb[df_celeb['label'] == 'FAKE'])}")
else:
    print("⚠️  Nenhum vídeo encontrado no Celeb-DF")

print()

# ============================================================================
# LER LIST OF TESTING VIDEOS (CELEB-DF)
# ============================================================================

list_file = f"{CELEB_BASE}/List_of_testing_videos.txt"
test_videos_celeb = []

if os.path.exists(list_file):
    print("📋 Lendo List_of_testing_videos.txt...")
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Formato esperado: "1 id5_id2_0000.mp4" ou "0 051.mp4"
                parts = line.split()
                if len(parts) >= 2:
                    label_num = parts[0]
                    video_name = parts[1]
                    test_videos_celeb.append(video_name)
    
    print(f"✓ {len(test_videos_celeb)} vídeos marcados como TEST no Celeb-DF")
    print()

# ============================================================================
# SALVAR ÍNDICES
# ============================================================================

print("💾 Salvando índices...")
print("-" * 80)

# Índice FaceForensics++
ff_output = f"{OUTPUT_DIR}/index_faceforensicspp.csv"
df_ff[['video_path', 'label', 'category', 'dataset', 'num_frames']].to_csv(
    ff_output, index=False
)
print(f"✓ FaceForensics++: {ff_output} ({len(df_ff)} vídeos)")

# Índice Celeb-DF
if len(df_celeb) > 0:
    celeb_output = f"{OUTPUT_DIR}/index_celebdf.csv"
    df_celeb.to_csv(celeb_output, index=False)
    print(f"✓ Celeb-DF: {celeb_output} ({len(df_celeb)} vídeos)")

# Índice combinado
df_combined = pd.concat([df_ff[['video_path', 'label', 'category', 'dataset']], 
                         df_celeb[['video_path', 'label', 'category', 'dataset']]], 
                        ignore_index=True)

combined_output = f"{OUTPUT_DIR}/index_all_datasets.csv"
df_combined.to_csv(combined_output, index=False)
print(f"✓ Combinado: {combined_output} ({len(df_combined)} vídeos)")

print()

# ============================================================================
# ESTATÍSTICAS
# ============================================================================

print("=" * 80)
print("📊 ESTATÍSTICAS FINAIS")
print("=" * 80)
print()

print("FaceForensics++:")
print(f"  Total: {len(df_ff)}")
for cat in df_ff['category'].unique():
    cat_df = df_ff[df_ff['category'] == cat]
    print(f"    {cat:20s}: {len(cat_df):4d} ({len(cat_df[cat_df['label']=='REAL'])} REAL, {len(cat_df[cat_df['label']=='FAKE'])} FAKE)")

print()

if len(df_celeb) > 0:
    print("Celeb-DF:")
    print(f"  Total: {len(df_celeb)}")
    for cat in df_celeb['category'].unique():
        cat_df = df_celeb[df_celeb['category'] == cat]
        print(f"    {cat:20s}: {len(cat_df):4d}")
    
    if test_videos_celeb:
        print(f"\n  Vídeos de teste marcados: {len(test_videos_celeb)}")

print()
print("TOTAL GERAL:")
print(f"  Vídeos: {len(df_combined)}")
print(f"  REAL: {len(df_combined[df_combined['label'] == 'REAL'])}")
print(f"  FAKE: {len(df_combined[df_combined['label'] == 'FAKE'])}")
print()

print("=" * 80)
print("✅ PROCESSAMENTO CONCLUÍDO!")
print("=" * 80)
print()
print("Próximos passos:")
print("  1. python scripts/generate_splits.py  # Gerar train/val/test splits")
print("  2. python train_full.py                # Treinar modelo")
print()
