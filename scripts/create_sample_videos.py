"""
Script auxiliar para criar vídeos de exemplo para demonstração.
Útil para testar o pipeline sem os datasets completos.
"""

import os
import cv2
import numpy as np


def create_sample_video(output_path, duration_sec=2, fps=30, resolution=(224, 224)):
    """
    Cria um vídeo de exemplo simples.
    
    Args:
        output_path (str): Caminho do vídeo de saída
        duration_sec (int): Duração em segundos
        fps (int): Frames por segundo
        resolution (tuple): Resolução (width, height)
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    num_frames = duration_sec * fps
    
    for i in range(num_frames):
        # Criar frame com gradiente de cor variável
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Gradiente baseado no frame atual
        color_value = int((i / num_frames) * 255)
        frame[:, :] = [color_value, 128, 255 - color_value]
        
        # Adicionar texto com número do frame
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"  ✓ Criado: {os.path.basename(output_path)} ({num_frames} frames)")


def main():
    """Função principal."""
    print("=" * 60)
    print("CRIAÇÃO DE VÍDEOS DE EXEMPLO")
    print("=" * 60)
    print()
    
    data_root = os.path.join(os.path.dirname(__file__), 'data')
    
    # FaceForensics++
    ff_real = os.path.join(data_root, 'FaceForensics++', 'videos_real')
    ff_fake = os.path.join(data_root, 'FaceForensics++', 'videos_fake')
    
    print("Criando vídeos de exemplo para FaceForensics++...\n")
    print("Videos reais:")
    for i in range(10):
        video_path = os.path.join(ff_real, f'real_sample_{i:03d}.mp4')
        create_sample_video(video_path, duration_sec=2, fps=30)
    
    print("\nVideos fake:")
    for i in range(10):
        video_path = os.path.join(ff_fake, f'fake_sample_{i:03d}.mp4')
        create_sample_video(video_path, duration_sec=2, fps=30)
    
    # Celeb-DF-v2 (menos vídeos para teste cross-dataset)
    celeb_real = os.path.join(data_root, 'Celeb-DF-v2', 'videos_real')
    celeb_fake = os.path.join(data_root, 'Celeb-DF-v2', 'videos_fake')
    
    print("\n\nCriando vídeos de exemplo para Celeb-DF-v2...\n")
    print("Videos reais:")
    for i in range(5):
        video_path = os.path.join(celeb_real, f'celeb_real_{i:03d}.mp4')
        create_sample_video(video_path, duration_sec=2, fps=30)
    
    print("\nVideos fake:")
    for i in range(5):
        video_path = os.path.join(celeb_fake, f'celeb_fake_{i:03d}.mp4')
        create_sample_video(video_path, duration_sec=2, fps=30)
    
    # WildDeepfake
    wild_real = os.path.join(data_root, 'WildDeepfake', 'videos_real')
    wild_fake = os.path.join(data_root, 'WildDeepfake', 'videos_fake')
    
    print("\n\nCriando vídeos de exemplo para WildDeepfake...\n")
    print("Videos reais:")
    for i in range(5):
        video_path = os.path.join(wild_real, f'wild_real_{i:03d}.mp4')
        create_sample_video(video_path, duration_sec=2, fps=30)
    
    print("\nVideos fake:")
    for i in range(5):
        video_path = os.path.join(wild_fake, f'wild_fake_{i:03d}.mp4')
        create_sample_video(video_path, duration_sec=2, fps=30)
    
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"\n✓ Total criado: 30 vídeos de exemplo")
    print(f"  - FaceForensics++: 20 vídeos (10 reais, 10 fakes)")
    print(f"  - Celeb-DF-v2: 10 vídeos (5 reais, 5 fakes)")
    print(f"  - WildDeepfake: 10 vídeos (5 reais, 5 fakes)")
    print(f"\nPróximos passos:")
    print(f"  1. python organize_datasets.py")
    print(f"  2. python generate_splits.py")
    print()


if __name__ == '__main__':
    main()
