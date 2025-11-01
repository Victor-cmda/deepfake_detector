"""
Script de treinamento do modelo DeepfakeDetector.
Inclui loop de treinamento, validação e early stopping.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import set_global_seed, get_device
from src.model import create_model, save_model, load_model
from src.preprocessing import get_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """
    Executa uma época de treinamento com suporte a Mixed Precision (AMP).
    
    Args:
        model: Modelo DeepfakeDetector
        dataloader: DataLoader de treino
        criterion: Função de perda (BCEWithLogitsLoss)
        optimizer: Otimizador (Adam)
        device: Dispositivo (cpu/cuda/mps)
        scaler: GradScaler para mixed precision (None desabilita AMP)
        
    Returns:
        avg_loss: Perda média da época
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None and device.type == 'cuda'
    
    for videos, labels in tqdm(dataloader, desc="Training", leave=False):
        # Move dados para o device
        videos = videos.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        # Zero gradientes
        optimizer.zero_grad()
        
        # Forward pass com Mixed Precision
        if use_amp:
            with autocast('cuda'):
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            # Backward pass com scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass normal
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            # Backward pass normal
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_epoch(model, dataloader, criterion, device):
    """
    Executa uma época de validação.
    
    Args:
        model: Modelo DeepfakeDetector
        dataloader: DataLoader de validação
        criterion: Função de perda (BCELoss)
        device: Dispositivo (cpu/cuda/mps)
        
    Returns:
        avg_loss: Perda média da validação
        f1: F1-score
        auc: AUC-ROC
        all_preds: Todas as predições (para análise)
        all_labels: Todos os labels verdadeiros
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc="Validation", leave=False):
            # Move dados para o device
            videos = videos.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            num_batches += 1
            
            # Coletar predições e labels (aplicar sigmoid para converter logits em probabilidades)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    
    # Calcular métricas
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # AUC requer pelo menos uma amostra de cada classe
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0
    
    return avg_loss, f1, auc, all_preds, all_labels


def train_model(
    splits_csv='data/splits_faceforensicspp.csv',
    batch_size=4,
    num_frames=16,
    num_epochs=20,
    learning_rate=1e-4,
    patience=5,
    model_save_path='models/model_best.pt',
    metrics_save_path='outputs/metrics_train.csv',
    early_stopping_log='outputs/logs/early_stopping.txt',
    device=None,
    num_workers=0
):
    """
    Função principal de treinamento do modelo DeepfakeDetector.
    
    Args:
        splits_csv: Caminho para o arquivo CSV com splits
        batch_size: Tamanho do batch
        num_frames: Número de frames por vídeo
        num_epochs: Número máximo de épocas
        learning_rate: Taxa de aprendizado inicial
        patience: Paciência para early stopping
        model_save_path: Caminho para salvar o melhor modelo
        metrics_save_path: Caminho para salvar métricas de treino
        early_stopping_log: Caminho para log de early stopping
        device: Dispositivo (None = auto-detect)
        num_workers: Workers para DataLoader
        
    Returns:
        model: Modelo treinado
        history: Histórico de métricas
    """
    # Configurar seed global
    set_global_seed(42)
    
    # Detectar device
    if device is None:
        device = get_device()
    
    print(f"\n{'='*60}")
    print(f"TREINAMENTO DO MODELO DEEPFAKE DETECTOR")
    print(f"{'='*60}\n")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Num epochs: {num_epochs}")
    print(f"Early stopping patience: {patience}")
    print(f"Splits CSV: {splits_csv}\n")
    
    # Criar diretórios de saída
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(early_stopping_log), exist_ok=True)
    
    # Criar DataLoaders
    print("Carregando datasets...")
    dataloaders = get_dataloaders(
        splits_csv=splits_csv,
        batch_size=batch_size,
        num_frames=num_frames,
        num_workers=num_workers,
        shuffle_train=True,
        cache_preprocessed=False
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Criar modelo (return_logits=True para usar com BCEWithLogitsLoss)
    print("Criando modelo...")
    model = create_model(num_frames=num_frames, pretrained=True, device=device, return_logits=True)
    model = model.to(device)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}\n")
    
    # Calcular class weights para balanceamento
    # Dataset: 700 REAL (14.3%) vs 4200 FAKE (85.7%)
    total_samples = len(train_loader.dataset)
    num_real = sum(1 for label in train_loader.dataset.labels if label == 0)
    num_fake = sum(1 for label in train_loader.dataset.labels if label == 1)
    
    # pos_weight = num_negatives / num_positives (para BCEWithLogitsLoss)
    # Quanto maior o pos_weight, mais o modelo penaliza erros em positivos (FAKE)
    # Como REAL é minoria, queremos penalizar mais erros em REAL
    # Logo: pos_weight = num_real / num_fake (inverso)
    pos_weight = torch.tensor([num_real / num_fake]).to(device)
    
    print(f"Class Balancing:")
    print(f"  REAL (0): {num_real} ({num_real/total_samples*100:.1f}%)")
    print(f"  FAKE (1): {num_fake} ({num_fake/total_samples*100:.1f}%)")
    print(f"  pos_weight: {pos_weight.item():.3f} (penaliza mais erros em FAKE)\n")
    
    # Definir função de perda com balanceamento e otimizador
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler: ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
    
    # Mixed Precision Training (AMP) - apenas para CUDA
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print("✓ Mixed Precision Training (FP16) ativado\n")
    
    # Variáveis para early stopping (usando AUC - mais robusto para desbalanceamento)
    best_val_auc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    # Histórico de métricas
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_auc': [],
        'learning_rate': []
    }
    
    # Loop de treinamento
    print(f"{'='*60}")
    print("INICIANDO TREINAMENTO")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        print(f"\nÉpoca {epoch}/{num_epochs}")
        print(f"{'-'*60}")
        
        # Treinar (com Mixed Precision se disponível)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validar
        val_loss, val_f1, val_auc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Atualizar scheduler
        scheduler.step(val_loss)
        
        # Obter learning rate atual
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Salvar métricas
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['learning_rate'].append(current_lr)
        
        # Imprimir métricas
        print(f"\nResultados:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val F1:     {val_f1:.4f}")
        print(f"  Val AUC:    {val_auc:.4f}")
        print(f"  LR:         {current_lr:.2e}")
        print(f"  Tempo:      {epoch_time:.2f}s")
        
        # Verificar se é o melhor modelo (usando AUC como métrica principal)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Salvar melhor modelo
            save_model(model, model_save_path)
            print(f"\n✓ Melhor modelo salvo! (AUC: {best_val_auc:.4f}, Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"\n  Sem melhoria no AUC por {epochs_no_improve} época(s)")
            print(f"  Melhor AUC: {best_val_auc:.4f} (época {best_epoch})")
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING ATIVADO (paciência: {patience})")
            print(f"{'='*60}")
            break
    
    total_time = time.time() - start_time
    
    # Salvar métricas em CSV
    df_metrics = pd.DataFrame(history)
    df_metrics.to_csv(metrics_save_path, index=False)
    print(f"\n✓ Métricas salvas em: {metrics_save_path}")
    
    # Salvar log de early stopping
    with open(early_stopping_log, 'w') as f:
        f.write("EARLY STOPPING LOG\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Melhor época: {best_epoch}\n")
        f.write(f"Melhor Val AUC: {best_val_auc:.4f}\n")
        f.write(f"Melhor Val Loss: {best_val_loss:.4f}\n")
        f.write(f"Paciência configurada: {patience}\n")
        f.write(f"Épocas sem melhoria: {epochs_no_improve}\n")
        f.write(f"Total de épocas executadas: {epoch}\n")
        f.write(f"Tempo total de treinamento: {total_time/60:.2f} min\n\n")
        f.write(f"Modelo salvo em: {model_save_path}\n")
        f.write(f"Métricas salvas em: {metrics_save_path}\n")
    
    print(f"✓ Early stopping log salvo em: {early_stopping_log}")
    
    # Sumário final
    print(f"\n{'='*60}")
    print("TREINAMENTO CONCLUÍDO")
    print(f"{'='*60}")
    print(f"Melhor época: {best_epoch}")
    print(f"Melhor Val AUC: {best_val_auc:.4f}")
    print(f"Melhor Val Loss: {best_val_loss:.4f}")
    print(f"Total de épocas: {epoch}")
    print(f"Tempo total: {total_time/60:.2f} min")
    print(f"{'='*60}\n")
    
    return model, history


def test_training():
    """
    Função de teste para validar o treinamento.
    Executa um treinamento rápido com poucos dados.
    """
    print("\n" + "="*60)
    print("TESTE DE TREINAMENTO")
    print("="*60 + "\n")
    
    # Verificar se existe CSV de splits
    splits_csv = 'data/splits_faceforensicspp.csv'
    if not os.path.exists(splits_csv):
        print(f"ERRO: Arquivo {splits_csv} não encontrado!")
        print("Execute primeiro as tarefas 2 e 3 para criar os datasets.")
        return
    
    # Treinar por poucas épocas
    model, history = train_model(
        splits_csv=splits_csv,
        batch_size=2,
        num_frames=16,
        num_epochs=5,
        learning_rate=1e-4,
        patience=3,
        num_workers=0
    )
    
    # Verificar outputs
    print("\nVerificando outputs gerados...")
    
    files_to_check = [
        'models/model_best.pt',
        'outputs/metrics_train.csv',
        'outputs/logs/early_stopping.txt'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"✗ {file_path} NÃO ENCONTRADO!")
    
    # Mostrar métricas finais
    print("\nMétricas finais:")
    df = pd.read_csv('outputs/metrics_train.csv')
    print(df.tail())
    
    print("\n" + "="*60)
    print("TESTE CONCLUÍDO")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Executar teste de treinamento
    test_training()
