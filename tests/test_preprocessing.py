"""
Script de teste do módulo de pré-processamento.
Demonstra extração de frames, detecção facial e normalização.
"""

import os
import sys
import pandas as pd
from src.preprocessing import (
    extract_frames, preprocess_video, visualize_preprocessing,
    batch_preprocess_videos, create_preprocessing_report
)
from src.utils import set_global_seed, get_device, ensure_dir
from facenet_pytorch import MTCNN


def test_single_video(video_path, output_dir='outputs/figures'):
    """
    Testa pré-processamento em um único vídeo.
    
    Args:
        video_path (str): Caminho do vídeo
        output_dir (str): Diretório para salvar visualizações
    """
    print("=" * 60)
    print("TESTE DE PRÉ-PROCESSAMENTO - VÍDEO ÚNICO")
    print("=" * 60)
    print()
    
    ensure_dir(output_dir)
    
    # Configurar device
    device = get_device()
    
    # Inicializar MTCNN
    print("\nInicializando MTCNN...")
    mtcnn = MTCNN(
        image_size=224,
        margin=0,
        keep_all=False,
        device='cpu',  # Usar CPU para compatibilidade
        post_process=False
    )
    print("✓ MTCNN inicializado")
    
    # Testar extração de frames
    print(f"\n1. Extraindo frames de: {os.path.basename(video_path)}")
    frames = extract_frames(video_path, num_frames=16)
    
    if frames is not None:
        print(f"  ✓ {len(frames)} frames extraídos")
        print(f"  - Shape de cada frame: {frames[0].shape}")
    else:
        print("  ✗ Falha na extração de frames")
        return
    
    # Testar pré-processamento completo
    print(f"\n2. Aplicando pré-processamento completo...")
    result = preprocess_video(video_path, mtcnn, num_frames=16)
    
    if result is not None:
        video_tensor, detection_rate, proc_time = result
        print(f"  ✓ Pré-processamento concluído")
        print(f"  - Shape do tensor: {video_tensor.shape}")
        print(f"  - Taxa de detecção facial: {detection_rate:.1f}%")
        print(f"  - Tempo de processamento: {proc_time:.2f}s")
    else:
        print("  ✗ Falha no pré-processamento")
        return
    
    # Criar visualização
    print(f"\n3. Gerando visualização...")
    output_path = os.path.join(output_dir, 'preprocessing_example.png')
    visualize_preprocessing(video_path, mtcnn, output_path)
    
    print("\n✓ Teste concluído com sucesso!")
    print()


def test_batch_processing():
    """
    Testa pré-processamento em lote com vários vídeos.
    """
    print("=" * 60)
    print("TESTE DE PRÉ-PROCESSAMENTO - LOTE")
    print("=" * 60)
    print()
    
    # Carregar alguns vídeos do dataset
    data_root = os.path.join(os.path.dirname(__file__), 'data')
    splits_file = os.path.join(data_root, 'splits_faceforensicspp.csv')
    
    if not os.path.exists(splits_file):
        print(f"✗ Arquivo não encontrado: {splits_file}")
        print("Execute primeiro: python generate_splits.py")
        return
    
    df = pd.read_csv(splits_file)
    
    if len(df) == 0:
        print("✗ Nenhum vídeo encontrado no arquivo de splits")
        return
    
    # Selecionar alguns vídeos para teste (2 reais, 2 fakes)
    df_real = df[df['label'] == 0].head(2)
    df_fake = df[df['label'] == 1].head(2)
    df_test = pd.concat([df_real, df_fake])
    
    video_paths = df_test['video_path'].tolist()
    
    print(f"Testando com {len(video_paths)} vídeos:")
    for i, path in enumerate(video_paths, 1):
        label = "REAL" if df_test.iloc[i-1]['label'] == 0 else "FAKE"
        print(f"  {i}. {os.path.basename(path)} ({label})")
    
    # Configurar device
    device = get_device()
    
    # Processar em lote
    stats = batch_preprocess_videos(video_paths, device=device, num_frames=16)
    
    # Mostrar estatísticas
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS DE PROCESSAMENTO")
    print("=" * 60)
    print(f"\nTotal de vídeos: {stats['total_videos']}")
    print(f"Processados com sucesso: {stats['processed']}")
    print(f"Falhas: {stats['failed']}")
    print(f"\nTempo total: {stats['total_time']:.2f}s")
    if stats['processed'] > 0:
        print(f"Tempo médio por vídeo: {stats['total_time']/stats['processed']:.2f}s")
    print(f"\nTaxa de detecção facial média: {stats['avg_detection_rate']:.1f}%")
    
    # Criar relatório
    create_preprocessing_report(stats)
    
    print("\n✓ Teste em lote concluído!")
    print()


def test_preprocessing_pipeline():
    """
    Testa pipeline completo de pré-processamento.
    """
    print("=" * 60)
    print("TESTE COMPLETO DO PIPELINE DE PRÉ-PROCESSAMENTO")
    print("=" * 60)
    print()
    
    # Configurar seed
    set_global_seed(42)
    
    # Encontrar um vídeo de exemplo
    data_root = os.path.join(os.path.dirname(__file__), 'data')
    ff_real = os.path.join(data_root, 'FaceForensics++', 'videos_real')
    
    video_files = [f for f in os.listdir(ff_real) if f.endswith('.mp4')]
    
    if not video_files:
        print("✗ Nenhum vídeo encontrado para teste")
        print("Execute primeiro: python create_sample_videos.py")
        return
    
    video_path = os.path.join(ff_real, video_files[0])
    
    # Teste 1: Vídeo único
    test_single_video(video_path)
    
    # Teste 2: Processamento em lote
    test_batch_processing()
    
    print("=" * 60)
    print("TODOS OS TESTES CONCLUÍDOS")
    print("=" * 60)
    print()
    print("Métricas registradas:")
    print("  ✓ Taxa de detecção facial")
    print("  ✓ Tempo de processamento")
    print("  ✓ Visualizações geradas")
    print()


def main():
    """Função principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Testar pré-processamento')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'full'], 
                       default='full', help='Modo de teste')
    parser.add_argument('--video', type=str, help='Caminho do vídeo (modo single)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.video:
            print("✗ Especifique um vídeo com --video")
            return
        test_single_video(args.video)
    elif args.mode == 'batch':
        test_batch_processing()
    else:
        test_preprocessing_pipeline()


if __name__ == '__main__':
    main()
