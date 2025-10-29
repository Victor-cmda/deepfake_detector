"""
Script para organizar e indexar os datasets de deepfake.
Este script deve ser executado após copiar os vídeos para as pastas apropriadas.

Estrutura esperada:
data/
├── FaceForensics++/
│   ├── videos_real/  <- Colocar vídeos reais aqui
│   └── videos_fake/  <- Colocar vídeos fake aqui
├── Celeb-DF-v2/
│   ├── videos_real/
│   └── videos_fake/
└── WildDeepfake/
    ├── videos_real/
    └── videos_fake/

Outputs gerados:
- data/index_faceforensicspp.csv
- data/index_celebdf.csv
- data/index_wilddeepfake.csv
"""

import os
import sys
from src.utils import organize_all_datasets, set_global_seed


def main():
    """Função principal."""
    print("=" * 60)
    print("ORGANIZAÇÃO E INDEXAÇÃO DE DATASETS")
    print("=" * 60)
    print()
    
    # Configurar seed
    set_global_seed(42)
    
    # Diretório raiz dos dados
    data_root = os.path.join(os.path.dirname(__file__), 'data')
    
    print(f"Diretório de dados: {data_root}\n")
    
    # Verificar estrutura
    datasets_to_check = ['FaceForensics++', 'Celeb-DF-v2', 'WildDeepfake']
    print("Verificando estrutura de diretórios...\n")
    
    for dataset_name in datasets_to_check:
        dataset_path = os.path.join(data_root, dataset_name)
        real_path = os.path.join(dataset_path, 'videos_real')
        fake_path = os.path.join(dataset_path, 'videos_fake')
        
        if os.path.exists(dataset_path):
            print(f"✓ {dataset_name}:")
            print(f"  - videos_real/: {'Existe' if os.path.exists(real_path) else 'NÃO EXISTE'}")
            print(f"  - videos_fake/: {'Existe' if os.path.exists(fake_path) else 'NÃO EXISTE'}")
        else:
            print(f"✗ {dataset_name}: NÃO EXISTE")
        print()
    
    # Organizar e indexar
    print("Iniciando indexação dos datasets...\n")
    datasets = organize_all_datasets(data_root)
    
    # Resumo final
    print("=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    
    total_videos = 0
    total_real = 0
    total_fake = 0
    
    for name, df in datasets.items():
        total_videos += len(df)
        total_real += len(df[df['label'] == 0])
        total_fake += len(df[df['label'] == 1])
    
    print(f"\nTotal geral:")
    print(f"  - Datasets indexados: {len(datasets)}")
    print(f"  - Total de vídeos: {total_videos}")
    print(f"  - Vídeos reais: {total_real}")
    print(f"  - Vídeos fake: {total_fake}")
    print()
    
    if total_videos == 0:
        print("⚠ ATENÇÃO: Nenhum vídeo foi encontrado!")
        print("Por favor, copie os vídeos para as pastas apropriadas:")
        print("  - data/FaceForensics++/videos_real/")
        print("  - data/FaceForensics++/videos_fake/")
        print("  - data/Celeb-DF-v2/videos_real/")
        print("  - data/Celeb-DF-v2/videos_fake/")
        print("  - data/WildDeepfake/videos_real/")
        print("  - data/WildDeepfake/videos_fake/")
        print()
        print("Depois execute novamente: python organize_datasets.py")
    else:
        print("✓ Indexação concluída com sucesso!")
    
    print()


if __name__ == '__main__':
    main()
