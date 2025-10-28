"""
Script para criar visualizações comparativas de pré-processamento.
Mostra exemplos de vídeos reais e fake lado a lado.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import extract_frames, detect_and_crop_face
from src.utils import set_global_seed, ensure_dir
from facenet_pytorch import MTCNN


def create_comparison_visualization():
    """
    Cria visualização comparando pré-processamento de vídeos reais e fake.
    """
    print("=" * 60)
    print("CRIAÇÃO DE VISUALIZAÇÃO COMPARATIVA")
    print("=" * 60)
    print()
    
    set_global_seed(42)
    
    # Carregar splits
    data_root = os.path.join(os.path.dirname(__file__), 'data')
    splits_file = os.path.join(data_root, 'splits_faceforensicspp.csv')
    
    df = pd.read_csv(splits_file)
    
    # Selecionar 1 vídeo real e 1 fake
    video_real = df[df['label'] == 0].iloc[0]['video_path']
    video_fake = df[df['label'] == 1].iloc[0]['video_path']
    
    print(f"Vídeo REAL: {os.path.basename(video_real)}")
    print(f"Vídeo FAKE: {os.path.basename(video_fake)}")
    print()
    
    # Inicializar MTCNN
    print("Inicializando MTCNN...")
    mtcnn = MTCNN(
        image_size=224,
        margin=0,
        keep_all=False,
        device='cpu',
        post_process=False
    )
    
    # Extrair e processar frames
    print("\nProcessando vídeo REAL...")
    frames_real = extract_frames(video_real, num_frames=8)
    processed_real = [detect_and_crop_face(f, mtcnn) for f in frames_real]
    
    print("Processando vídeo FAKE...")
    frames_fake = extract_frames(video_fake, num_frames=8)
    processed_fake = [detect_and_crop_face(f, mtcnn) for f in frames_fake]
    
    # Criar visualização comparativa
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))
    fig.suptitle('Comparação: Pré-processamento de Vídeos Reais vs Fake', 
                 fontsize=16, fontweight='bold')
    
    # Linha 1: REAL
    for i in range(8):
        axes[0, i].imshow(processed_real[i])
        axes[0, i].set_title(f'Real F{i+1}', fontsize=10)
        axes[0, i].axis('off')
    
    # Linha 2: FAKE
    for i in range(8):
        axes[1, i].imshow(processed_fake[i])
        axes[1, i].set_title(f'Fake F{i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    # Labels nas laterais
    axes[0, 0].set_ylabel('REAL', fontsize=14, fontweight='bold', rotation=0, 
                          labelpad=40, va='center')
    axes[1, 0].set_ylabel('FAKE', fontsize=14, fontweight='bold', rotation=0, 
                          labelpad=40, va='center')
    
    plt.tight_layout()
    
    # Salvar
    output_path = 'outputs/figures/preprocessing_comparison.png'
    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualização salva: {output_path}")
    plt.close()
    
    print("\n✓ Comparação concluída!")
    print()


def main():
    """Função principal."""
    create_comparison_visualization()


if __name__ == '__main__':
    main()
