"""
Script para gerar divisões treino/validação/teste para os datasets.

Gera o arquivo data/splits_faceforensicspp.csv com colunas:
- video_path: Caminho do vídeo
- label: 0 (real) ou 1 (fake)
- dataset: Nome do dataset
- num_frames: Número de frames
- split: 'train', 'val' ou 'test'

Distribuição: 70% treino, 15% validação, 15% teste

Os datasets externos (Celeb-DF-v2 e WildDeepfake) permanecem completos
para avaliação cross-dataset.
"""

import os
import sys
from src.utils import generate_train_val_test_split, set_global_seed


def main():
    """Função principal."""
    print("=" * 60)
    print("GERAÇÃO DE DIVISÕES TREINO/VALIDAÇÃO/TESTE")
    print("=" * 60)
    print()
    
    # Configurar seed
    set_global_seed(42)
    
    # Diretório de dados
    data_root = os.path.join(os.path.dirname(__file__), 'data')
    
    # Arquivos de entrada e saída
    faceforensics_index = os.path.join(data_root, 'index_faceforensicspp.csv')
    faceforensics_splits = os.path.join(data_root, 'splits_faceforensicspp.csv')
    
    print("Configuração:")
    print(f"  - Seed: 42")
    print(f"  - Distribuição: 70% treino, 15% val, 15% teste")
    print(f"  - Dataset: FaceForensics++")
    print()
    
    # Verificar se arquivo de índice existe
    if not os.path.exists(faceforensics_index):
        print(f"✗ Arquivo de índice não encontrado: {faceforensics_index}")
        print()
        print("Por favor, execute primeiro: python organize_datasets.py")
        return
    
    print("Gerando divisões para FaceForensics++...\n")
    
    # Gerar divisão
    df_splits = generate_train_val_test_split(
        index_csv=faceforensics_index,
        output_csv=faceforensics_splits,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    if df_splits is None or len(df_splits) == 0:
        print("⚠ ATENÇÃO: Nenhum vídeo foi encontrado no índice!")
        print()
        print("Etapas necessárias:")
        print("  1. Copiar vídeos para data/FaceForensics++/videos_real/ e videos_fake/")
        print("  2. Executar: python organize_datasets.py")
        print("  3. Executar: python generate_splits.py")
        print()
    else:
        print("=" * 60)
        print("RESUMO FINAL")
        print("=" * 60)
        print()
        print(f"✓ Arquivo gerado: {faceforensics_splits}")
        print(f"✓ Total de vídeos: {len(df_splits)}")
        print()
        print("Observações:")
        print("  - Datasets externos (Celeb-DF-v2, WildDeepfake) permanecem completos")
        print("  - Estes serão usados para avaliação cross-dataset na Tarefa 9")
        print("  - A divisão é estratificada por label (mantém proporção real/fake)")
        print()
        print("✓ Divisões geradas com sucesso!")
    
    print()


if __name__ == '__main__':
    main()
