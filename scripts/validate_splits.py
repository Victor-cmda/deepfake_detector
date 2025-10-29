"""
Script de validação das divisões treino/validação/teste.
Verifica integridade, distribuição e estatísticas.
"""

import os
import pandas as pd
from src.utils import set_global_seed


def validate_splits(splits_csv):
    """
    Valida o arquivo de divisões.
    
    Args:
        splits_csv (str): Caminho do arquivo CSV de splits
    """
    if not os.path.exists(splits_csv):
        print(f"✗ Arquivo não encontrado: {splits_csv}")
        return False
    
    df = pd.read_csv(splits_csv)
    
    print(f"\n{'='*60}")
    print(f"VALIDAÇÃO: {os.path.basename(splits_csv)}")
    print(f"{'='*60}\n")
    
    # Verificar colunas
    required_cols = ['video_path', 'label', 'dataset', 'num_frames', 'split']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"✗ Colunas faltando: {missing_cols}")
        return False
    
    print(f"✓ Colunas presentes: {list(df.columns)}")
    
    # Verificar valores de split
    unique_splits = df['split'].unique()
    expected_splits = ['train', 'val', 'test']
    
    print(f"✓ Splits encontrados: {sorted(unique_splits)}")
    
    if not all(s in expected_splits for s in unique_splits):
        print(f"✗ Splits inválidos encontrados")
        return False
    
    # Estatísticas gerais
    print(f"\n{'ESTATÍSTICAS GERAIS':^60}")
    print(f"{'-'*60}")
    print(f"Total de vídeos: {len(df)}")
    print(f"Vídeos reais (label=0): {len(df[df['label']==0])}")
    print(f"Vídeos fake (label=1): {len(df[df['label']==1])}")
    print(f"Média de frames: {df['num_frames'].mean():.1f}")
    
    # Estatísticas por split
    print(f"\n{'DISTRIBUIÇÃO POR SPLIT':^60}")
    print(f"{'-'*60}")
    
    for split in ['train', 'val', 'test']:
        df_split = df[df['split'] == split]
        n_total = len(df_split)
        n_real = len(df_split[df_split['label'] == 0])
        n_fake = len(df_split[df_split['label'] == 1])
        pct = (n_total / len(df)) * 100
        
        print(f"\n{split.upper()}:")
        print(f"  Total: {n_total} ({pct:.1f}%)")
        print(f"  Reais: {n_real} ({n_real/n_total*100:.1f}%)")
        print(f"  Fakes: {n_fake} ({n_fake/n_total*100:.1f}%)")
        print(f"  Balanceamento: {'✓ OK' if abs(n_real - n_fake) <= 1 else '⚠ Desbalanceado'}")
    
    # Verificar arquivos existem
    print(f"\n{'VERIFICAÇÃO DE ARQUIVOS':^60}")
    print(f"{'-'*60}")
    
    missing_files = []
    for video_path in df['video_path']:
        if not os.path.exists(video_path):
            missing_files.append(video_path)
    
    if missing_files:
        print(f"✗ {len(missing_files)} arquivos não encontrados:")
        for f in missing_files[:5]:  # Mostrar apenas primeiros 5
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... e mais {len(missing_files)-5}")
        return False
    else:
        print(f"✓ Todos os {len(df)} arquivos existem")
    
    # Verificar duplicatas
    print(f"\n{'VERIFICAÇÃO DE DUPLICATAS':^60}")
    print(f"{'-'*60}")
    
    duplicates = df[df.duplicated(subset=['video_path'], keep=False)]
    if len(duplicates) > 0:
        print(f"✗ {len(duplicates)} entradas duplicadas encontradas")
        return False
    else:
        print(f"✓ Nenhuma duplicata encontrada")
    
    print(f"\n{'='*60}")
    print(f"✓ VALIDAÇÃO CONCLUÍDA COM SUCESSO")
    print(f"{'='*60}\n")
    
    return True


def main():
    """Função principal."""
    print("=" * 60)
    print("VALIDAÇÃO DE DIVISÕES")
    print("=" * 60)
    
    set_global_seed(42)
    
    data_root = os.path.join(os.path.dirname(__file__), 'data')
    splits_file = os.path.join(data_root, 'splits_faceforensicspp.csv')
    
    success = validate_splits(splits_file)
    
    if success:
        print("\n✓ Arquivo de splits válido e pronto para uso!")
    else:
        print("\n✗ Problemas encontrados no arquivo de splits.")
    
    print()


if __name__ == '__main__':
    main()
