"""
Gera divisões train/val/test para os datasets reais.
Usa estratificação para manter proporção de REAL/FAKE em cada split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("=" * 80)
print("GERAÇÃO DE SPLITS TRAIN/VAL/TEST")
print("=" * 80)
print()

# Configurações
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

np.random.seed(RANDOM_SEED)

# ============================================================================
# FACEFORENSICS++ (Dataset principal para treinamento)
# ============================================================================

print("📊 Processando FaceForensics++...")
print("-" * 80)

df_ff = pd.read_csv('data/index_faceforensicspp.csv')

print(f"Total: {len(df_ff)} vídeos")
print(f"  REAL: {len(df_ff[df_ff['label'] == 'REAL'])}")
print(f"  FAKE: {len(df_ff[df_ff['label'] == 'FAKE'])}")
print()

# Criar divisões estratificadas (mantém proporção REAL/FAKE)
# Primeiro: train vs (val+test)
train_ff, temp_ff = train_test_split(
    df_ff,
    test_size=(VAL_RATIO + TEST_RATIO),
    random_state=RANDOM_SEED,
    stratify=df_ff['label']
)

# Segundo: val vs test
val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
val_ff, test_ff = train_test_split(
    temp_ff,
    test_size=(1 - val_ratio_adjusted),
    random_state=RANDOM_SEED,
    stratify=temp_ff['label']
)

# Adicionar coluna de split
train_ff['split'] = 'train'
val_ff['split'] = 'val'
test_ff['split'] = 'test'

# Combinar
df_ff_splits = pd.concat([train_ff, val_ff, test_ff], ignore_index=True)

print("Divisões FaceForensics++:")
print(f"  Train: {len(train_ff)} vídeos ({len(train_ff[train_ff['label']=='REAL'])} REAL, {len(train_ff[train_ff['label']=='FAKE'])} FAKE)")
print(f"  Val:   {len(val_ff)} vídeos ({len(val_ff[val_ff['label']=='REAL'])} REAL, {len(val_ff[val_ff['label']=='FAKE'])} FAKE)")
print(f"  Test:  {len(test_ff)} vídeos ({len(test_ff[test_ff['label']=='REAL'])} REAL, {len(test_ff[test_ff['label']=='FAKE'])} FAKE)")
print()

# Salvar
output_ff = 'data/splits_faceforensicspp.csv'
df_ff_splits.to_csv(output_ff, index=False)
print(f"✓ Salvo: {output_ff}")
print()

# ============================================================================
# CELEB-DF (Para avaliação cross-dataset)
# ============================================================================

print("📊 Processando Celeb-DF...")
print("-" * 80)

df_celeb = pd.read_csv('data/index_celebdf.csv')

print(f"Total: {len(df_celeb)} vídeos")
print(f"  REAL: {len(df_celeb[df_celeb['label'] == 'REAL'])}")
print(f"  FAKE: {len(df_celeb[df_celeb['label'] == 'FAKE'])}")
print()

# Ler lista de vídeos de teste oficial (se existir)
test_videos_file = 'E:/Celeb/List_of_testing_videos.txt'
test_videos_list = []

if os.path.exists(test_videos_file):
    print("📋 Usando List_of_testing_videos.txt para split de teste...")
    with open(test_videos_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    video_name = parts[1]
                    test_videos_list.append(video_name)
    print(f"  {len(test_videos_list)} vídeos marcados como TEST")
    print()

# Marcar vídeos de teste
if test_videos_list:
    df_celeb['is_test'] = df_celeb['video_name'].apply(lambda x: x in test_videos_list)
    test_celeb = df_celeb[df_celeb['is_test']]
    non_test_celeb = df_celeb[~df_celeb['is_test']]
    
    # Dividir non-test em train e val
    train_celeb, val_celeb = train_test_split(
        non_test_celeb,
        test_size=0.2,  # 20% para val
        random_state=RANDOM_SEED,
        stratify=non_test_celeb['label']
    )
else:
    # Sem lista oficial, usar divisão padrão
    print("⚠️  List_of_testing_videos.txt não encontrado, usando divisão padrão...")
    train_celeb, temp_celeb = train_test_split(
        df_celeb,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=df_celeb['label']
    )
    
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_celeb, test_celeb = train_test_split(
        temp_celeb,
        test_size=(1 - val_ratio_adjusted),
        random_state=RANDOM_SEED,
        stratify=temp_celeb['label']
    )

# Adicionar coluna de split
train_celeb['split'] = 'train'
val_celeb['split'] = 'val'
test_celeb['split'] = 'test'

# Combinar
df_celeb_splits = pd.concat([train_celeb, val_celeb, test_celeb], ignore_index=True)

print("Divisões Celeb-DF:")
print(f"  Train: {len(train_celeb)} vídeos ({len(train_celeb[train_celeb['label']=='REAL'])} REAL, {len(train_celeb[train_celeb['label']=='FAKE'])} FAKE)")
print(f"  Val:   {len(val_celeb)} vídeos ({len(val_celeb[val_celeb['label']=='REAL'])} REAL, {len(val_celeb[val_celeb['label']=='FAKE'])} FAKE)")
print(f"  Test:  {len(test_celeb)} vídeos ({len(test_celeb[test_celeb['label']=='REAL'])} REAL, {len(test_celeb[test_celeb['label']=='FAKE'])} FAKE)")
print()

# Salvar
output_celeb = 'data/splits_celebdf.csv'
df_celeb_splits.to_csv(output_celeb, index=False)
print(f"✓ Salvo: {output_celeb}")
print()

# ============================================================================
# ESTATÍSTICAS FINAIS
# ============================================================================

print("=" * 80)
print("📊 RESUMO DAS DIVISÕES")
print("=" * 80)
print()

print("FaceForensics++ (Treinamento Principal):")
print(f"  Total: {len(df_ff_splits)}")
for split in ['train', 'val', 'test']:
    split_df = df_ff_splits[df_ff_splits['split'] == split]
    pct = len(split_df) / len(df_ff_splits) * 100
    real_count = len(split_df[split_df['label'] == 'REAL'])
    fake_count = len(split_df[split_df['label'] == 'FAKE'])
    print(f"    {split.upper():5s}: {len(split_df):4d} ({pct:5.1f}%) - {real_count} REAL, {fake_count} FAKE")

print()
print("Celeb-DF (Avaliação Cross-Dataset):")
print(f"  Total: {len(df_celeb_splits)}")
for split in ['train', 'val', 'test']:
    split_df = df_celeb_splits[df_celeb_splits['split'] == split]
    pct = len(split_df) / len(df_celeb_splits) * 100
    real_count = len(split_df[split_df['label'] == 'REAL'])
    fake_count = len(split_df[split_df['label'] == 'FAKE'])
    print(f"    {split.upper():5s}: {len(split_df):4d} ({pct:5.1f}%) - {real_count} REAL, {fake_count} FAKE")

print()
print("=" * 80)
print("✅ DIVISÕES CRIADAS COM SUCESSO!")
print("=" * 80)
print()
print("Arquivos gerados:")
print(f"  • {output_ff}")
print(f"  • {output_celeb}")
print()
print("Próximo passo:")
print("  python train_full.py  # Treinar modelo com FaceForensics++")
print()
