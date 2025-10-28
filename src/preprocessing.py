"""
Módulo de pré-processamento de vídeos para detecção de deepfakes.
Inclui extração de frames, detecção facial e normalização.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# Configurações globais
FRAME_SIZE = 224  # Tamanho do frame após redimensionamento
FRAMES_PER_VIDEO = 16  # Número de frames a extrair por vídeo


def extract_frames(video_path, num_frames=FRAMES_PER_VIDEO, target_size=(FRAME_SIZE, FRAME_SIZE)):
    """
    Extrai frames uniformemente espaçados de um vídeo.
    
    Args:
        video_path (str): Caminho do vídeo
        num_frames (int): Número de frames a extrair
        target_size (tuple): Tamanho alvo (width, height)
        
    Returns:
        list: Lista de frames (numpy arrays) ou None se erro
    """
    if not os.path.exists(video_path):
        print(f"  ✗ Vídeo não encontrado: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"  ✗ Vídeo vazio: {video_path}")
        cap.release()
        return None
    
    # Calcular índices dos frames a extrair (uniformemente espaçados)
    if total_frames < num_frames:
        # Se o vídeo tem menos frames que o desejado, usar todos
        frame_indices = list(range(total_frames))
    else:
        # Extrair frames uniformemente espaçados
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Converter BGR para RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Redimensionar
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
    
    cap.release()
    
    # Se não conseguiu extrair frames suficientes, replicar o último
    while len(frames) < num_frames:
        if len(frames) > 0:
            frames.append(frames[-1].copy())
        else:
            # Criar frame preto se nenhum frame foi extraído
            frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
    
    return frames


def detect_and_crop_face(frame, mtcnn, margin=20):
    """
    Detecta face em um frame e retorna região recortada.
    
    Args:
        frame (numpy.ndarray): Frame de entrada (RGB)
        mtcnn (MTCNN): Detector MTCNN
        margin (int): Margem ao redor da face detectada
        
    Returns:
        numpy.ndarray: Face recortada ou frame original se não detectar
    """
    # Converter para PIL Image
    pil_image = Image.fromarray(frame)
    
    # Detectar face
    boxes, probs = mtcnn.detect(pil_image)
    
    if boxes is not None and len(boxes) > 0:
        # Pegar primeira face detectada (maior probabilidade)
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Adicionar margem
        h, w = frame.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Recortar face
        face = frame[y1:y2, x1:x2]
        
        # Redimensionar para tamanho padrão
        face = cv2.resize(face, (FRAME_SIZE, FRAME_SIZE))
        
        return face
    else:
        # Se não detectou face, retornar frame original redimensionado
        return cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))


def preprocess_video(video_path, mtcnn, num_frames=FRAMES_PER_VIDEO):
    """
    Pipeline completo de pré-processamento de vídeo.
    
    Args:
        video_path (str): Caminho do vídeo
        mtcnn (MTCNN): Detector MTCNN
        num_frames (int): Número de frames a extrair
        
    Returns:
        torch.Tensor: Tensor de frames processados (T, C, H, W) ou None
    """
    start_time = time.time()
    
    # Extrair frames
    frames = extract_frames(video_path, num_frames)
    
    if frames is None or len(frames) == 0:
        return None
    
    # Detectar e recortar faces
    processed_frames = []
    faces_detected = 0
    
    for frame in frames:
        face = detect_and_crop_face(frame, mtcnn)
        processed_frames.append(face)
        
        # Verificar se face foi detectada (comparando se mudou)
        if face.shape == (FRAME_SIZE, FRAME_SIZE, 3):
            faces_detected += 1
    
    # Normalização
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converte para tensor e normaliza [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Aplicar transformações
    tensor_frames = []
    for frame in processed_frames:
        frame_tensor = transform(frame)
        tensor_frames.append(frame_tensor)
    
    # Empilhar frames (T, C, H, W)
    video_tensor = torch.stack(tensor_frames)
    
    processing_time = time.time() - start_time
    detection_rate = faces_detected / len(frames) * 100
    
    return video_tensor, detection_rate, processing_time


def visualize_preprocessing(video_path, mtcnn, output_path=None):
    """
    Visualiza o resultado do pré-processamento de um vídeo.
    
    Args:
        video_path (str): Caminho do vídeo
        mtcnn (MTCNN): Detector MTCNN
        output_path (str): Caminho para salvar imagem (opcional)
    """
    # Extrair frames
    frames = extract_frames(video_path, num_frames=8)
    
    if frames is None or len(frames) == 0:
        print(f"  ✗ Não foi possível extrair frames de {video_path}")
        return
    
    # Processar frames
    processed_frames = []
    for frame in frames:
        face = detect_and_crop_face(frame, mtcnn)
        processed_frames.append(face)
    
    # Criar visualização
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Pré-processamento: {os.path.basename(video_path)}', fontsize=14, fontweight='bold')
    
    for idx in range(min(8, len(processed_frames))):
        row = idx // 4
        col = idx % 4
        
        axes[row, col].imshow(processed_frames[idx])
        axes[row, col].set_title(f'Frame {idx+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualização salva: {output_path}")
    else:
        plt.show()
    
    plt.close()


def batch_preprocess_videos(video_paths, device='cpu', num_frames=FRAMES_PER_VIDEO):
    """
    Pré-processa múltiplos vídeos em lote.
    
    Args:
        video_paths (list): Lista de caminhos de vídeos
        device (str): Device para MTCNN ('cpu', 'cuda', 'mps')
        num_frames (int): Número de frames por vídeo
        
    Returns:
        dict: Dicionário com estatísticas de processamento
    """
    # Inicializar MTCNN - usar sempre CPU para compatibilidade
    # (MTCNN tem issues com MPS no PyTorch atual)
    mtcnn = MTCNN(
        image_size=FRAME_SIZE,
        margin=0,
        keep_all=False,
        device='cpu',  # Sempre usar CPU para MTCNN
        post_process=False
    )
    
    stats = {
        'total_videos': len(video_paths),
        'processed': 0,
        'failed': 0,
        'total_time': 0,
        'avg_detection_rate': 0,
        'detection_rates': []
    }
    
    print(f"\nProcessando {len(video_paths)} vídeos...\n")
    
    for video_path in tqdm(video_paths, desc="Pré-processamento"):
        result = preprocess_video(video_path, mtcnn, num_frames)
        
        if result is not None:
            video_tensor, detection_rate, proc_time = result
            stats['processed'] += 1
            stats['total_time'] += proc_time
            stats['detection_rates'].append(detection_rate)
        else:
            stats['failed'] += 1
    
    # Calcular estatísticas finais
    if stats['detection_rates']:
        stats['avg_detection_rate'] = np.mean(stats['detection_rates'])
    
    return stats


def create_preprocessing_report(stats, output_file='outputs/logs/preprocessing_stats.txt'):
    """
    Cria relatório de estatísticas de pré-processamento.
    
    Args:
        stats (dict): Dicionário com estatísticas
        output_file (str): Caminho do arquivo de saída
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO DE PRÉ-PROCESSAMENTO\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total de vídeos: {stats['total_videos']}\n")
        f.write(f"Processados com sucesso: {stats['processed']}\n")
        f.write(f"Falhas: {stats['failed']}\n\n")
        
        f.write(f"Tempo total: {stats['total_time']:.2f}s\n")
        if stats['processed'] > 0:
            f.write(f"Tempo médio por vídeo: {stats['total_time']/stats['processed']:.2f}s\n")
        
        f.write(f"\nTaxa de detecção facial média: {stats['avg_detection_rate']:.1f}%\n")
        
        if stats['detection_rates']:
            f.write(f"Taxa mínima: {min(stats['detection_rates']):.1f}%\n")
            f.write(f"Taxa máxima: {max(stats['detection_rates']):.1f}%\n")
    
    print(f"\n✓ Relatório salvo: {output_file}")


# ============================================================================
# DATASET E DATALOADER (Tarefa 6)
# ============================================================================

class VideoDataset(Dataset):
    """
    Dataset customizado para vídeos de detecção de deepfakes.
    
    Carrega vídeos, aplica pré-processamento e retorna tensores prontos para treino.
    """
    
    def __init__(self, video_paths, labels, num_frames=FRAMES_PER_VIDEO, 
                 transform=None, device='cpu', cache_preprocessed=False):
        """
        Inicializa o dataset.
        
        Args:
            video_paths (list): Lista de caminhos dos vídeos
            labels (list): Lista de labels (0=real, 1=fake)
            num_frames (int): Número de frames por vídeo
            transform (callable): Transformações adicionais (opcional)
            device (str): Device para MTCNN
            cache_preprocessed (bool): Se True, cacheia vídeos pré-processados
        """
        assert len(video_paths) == len(labels), "Número de vídeos e labels deve ser igual"
        
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        self.device = device
        self.cache_preprocessed = cache_preprocessed
        
        # Inicializar MTCNN (sempre CPU para compatibilidade)
        self.mtcnn = MTCNN(
            image_size=FRAME_SIZE,
            margin=0,
            keep_all=False,
            device='cpu',
            post_process=False
        )
        
        # Cache para vídeos pré-processados
        self.cache = {} if cache_preprocessed else None
        
        print(f"✓ VideoDataset inicializado:")
        print(f"  - Total de vídeos: {len(self.video_paths)}")
        print(f"  - Reais: {sum(1 for l in labels if l == 0)}")
        print(f"  - Fakes: {sum(1 for l in labels if l == 1)}")
        print(f"  - Frames por vídeo: {num_frames}")
        print(f"  - Cache ativado: {cache_preprocessed}")
    
    def __len__(self):
        """Retorna o tamanho do dataset."""
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Retorna um item do dataset.
        
        Args:
            idx (int): Índice do item
            
        Returns:
            tuple: (video_tensor, label)
                - video_tensor: torch.Tensor (T, C, H, W)
                - label: int (0 ou 1)
        """
        # Verificar cache
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Pré-processar vídeo
        result = preprocess_video(video_path, self.mtcnn, self.num_frames)
        
        if result is None:
            # Se falhar, criar tensor zeros como fallback
            print(f"  ⚠ Falha ao processar {os.path.basename(video_path)}, usando zeros")
            video_tensor = torch.zeros(self.num_frames, 3, FRAME_SIZE, FRAME_SIZE)
        else:
            video_tensor, _, _ = result
        
        # Aplicar transformações adicionais se fornecidas
        if self.transform is not None:
            video_tensor = self.transform(video_tensor)
        
        # Converter label para tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        # Cachear se ativado
        if self.cache is not None:
            self.cache[idx] = (video_tensor, label_tensor)
        
        return video_tensor, label_tensor
    
    def get_labels(self):
        """Retorna lista de labels."""
        return self.labels
    
    def get_class_weights(self):
        """
        Calcula pesos das classes para balanceamento.
        
        Returns:
            torch.Tensor: Pesos das classes [weight_real, weight_fake]
        """
        num_real = sum(1 for l in self.labels if l == 0)
        num_fake = sum(1 for l in self.labels if l == 1)
        total = len(self.labels)
        
        weight_real = total / (2 * num_real) if num_real > 0 else 1.0
        weight_fake = total / (2 * num_fake) if num_fake > 0 else 1.0
        
        return torch.tensor([weight_real, weight_fake])


def collate_fn(batch):
    """
    Função de collate customizada para DataLoader.
    
    Agrupa vídeos em batches, tratando possíveis falhas de carregamento.
    
    Args:
        batch (list): Lista de tuplas (video_tensor, label)
        
    Returns:
        tuple: (videos_batch, labels_batch)
            - videos_batch: torch.Tensor (B, T, C, H, W)
            - labels_batch: torch.Tensor (B,)
    """
    # Filtrar itens None (se houver falhas)
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None, None
    
    # Separar vídeos e labels
    videos = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Empilhar em batch
    videos_batch = torch.stack(videos, dim=0)  # (B, T, C, H, W)
    labels_batch = torch.stack(labels, dim=0)  # (B,)
    
    return videos_batch, labels_batch


def get_dataloaders(splits_csv, batch_size=4, num_frames=FRAMES_PER_VIDEO, 
                   num_workers=0, shuffle_train=True, cache_preprocessed=False):
    """
    Cria DataLoaders para treino, validação e teste.
    
    Args:
        splits_csv (str): Caminho do arquivo CSV com splits
        batch_size (int): Tamanho do batch
        num_frames (int): Número de frames por vídeo
        num_workers (int): Número de workers para carregamento paralelo
        shuffle_train (bool): Embaralhar dados de treino
        cache_preprocessed (bool): Cachear vídeos pré-processados
        
    Returns:
        dict: Dicionário com DataLoaders {'train', 'val', 'test'}
    """
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("CRIAÇÃO DE DATALOADERS")
    print("=" * 60)
    print()
    
    # Carregar splits
    if not os.path.exists(splits_csv):
        raise FileNotFoundError(f"Arquivo não encontrado: {splits_csv}")
    
    df = pd.read_csv(splits_csv)
    print(f"✓ Splits carregados de: {splits_csv}")
    print(f"  Total de vídeos: {len(df)}")
    
    # Verificar colunas necessárias
    required_cols = ['video_path', 'label', 'split']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no CSV")
    
    dataloaders = {}
    
    for split_name in ['train', 'val', 'test']:
        df_split = df[df['split'] == split_name]
        
        if len(df_split) == 0:
            print(f"\n⚠ Split '{split_name}' vazio, pulando...")
            continue
        
        print(f"\n{split_name.upper()}:")
        print("-" * 60)
        
        # Extrair paths e labels
        video_paths = df_split['video_path'].tolist()
        labels = df_split['label'].tolist()
        
        # Criar dataset
        dataset = VideoDataset(
            video_paths=video_paths,
            labels=labels,
            num_frames=num_frames,
            device='cpu',
            cache_preprocessed=cache_preprocessed
        )
        
        # Criar dataloader
        shuffle = shuffle_train if split_name == 'train' else False
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        dataloaders[split_name] = dataloader
        
        print(f"  ✓ DataLoader criado:")
        print(f"    - Batches: {len(dataloader)}")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Shuffle: {shuffle}")
    
    print("\n" + "=" * 60)
    print(f"✓ {len(dataloaders)} DataLoaders criados com sucesso!")
    print("=" * 60)
    print()
    
    return dataloaders


def test_dataloader(dataloader, num_batches=2):
    """
    Testa um DataLoader carregando alguns batches.
    
    Args:
        dataloader (DataLoader): DataLoader a testar
        num_batches (int): Número de batches a testar
        
    Returns:
        dict: Estatísticas do teste
    """
    print(f"\nTestando DataLoader...")
    print(f"  - Total de batches disponíveis: {len(dataloader)}")
    print(f"  - Testando {min(num_batches, len(dataloader))} batches\n")
    
    stats = {
        'batches_tested': 0,
        'total_samples': 0,
        'total_time': 0,
        'shapes': [],
        'label_distribution': {'real': 0, 'fake': 0}
    }
    
    for i, (videos, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        
        start_time = time.time()
        
        if videos is None or labels is None:
            print(f"  ⚠ Batch {i+1}: Vazio (erro no carregamento)")
            continue
        
        batch_time = time.time() - start_time
        
        stats['batches_tested'] += 1
        stats['total_samples'] += len(videos)
        stats['total_time'] += batch_time
        stats['shapes'].append(videos.shape)
        
        # Contar labels
        for label in labels:
            if label.item() == 0:
                stats['label_distribution']['real'] += 1
            else:
                stats['label_distribution']['fake'] += 1
        
        print(f"  Batch {i+1}:")
        print(f"    - Videos shape: {videos.shape}")
        print(f"    - Labels shape: {labels.shape}")
        print(f"    - Labels: {labels.squeeze().tolist()}")
        print(f"    - Memory: {videos.element_size() * videos.nelement() / 1024 / 1024:.2f} MB")
        print(f"    - Tempo: {batch_time:.4f}s")
        print()
    
    # Resumo
    if stats['batches_tested'] > 0:
        avg_time = stats['total_time'] / stats['batches_tested']
        print(f"✓ Teste concluído:")
        print(f"  - Batches testados: {stats['batches_tested']}")
        print(f"  - Total de amostras: {stats['total_samples']}")
        print(f"  - Tempo médio/batch: {avg_time:.4f}s")
        print(f"  - Distribuição: {stats['label_distribution']['real']} reais, {stats['label_distribution']['fake']} fakes")
    
    return stats


