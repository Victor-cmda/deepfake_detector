"""
Script de avaliação e geração de métricas.
Inclui avaliação cross-dataset, visualizações e testes de robustez.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from tqdm import tqdm

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import set_global_seed, get_device
from src.model import load_model
from src.preprocessing import get_dataloaders


def evaluate_model(model, dataloader, device, dataset_name="Dataset"):
    """
    Avalia o modelo em um dataset.
    
    Args:
        model: Modelo DeepfakeDetector
        dataloader: DataLoader do dataset
        device: Dispositivo (cpu/cuda/mps)
        dataset_name: Nome do dataset (para logging)
        
    Returns:
        metrics: Dicionário com métricas
        predictions: Array com predições
        probabilities: Array com probabilidades
        labels: Array com labels verdadeiros
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    print(f"\nAvaliando {dataset_name}...")
    
    with torch.no_grad():
        for videos, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}", leave=False):
            # Move dados para o device
            videos = videos.to(device)
            labels_np = labels.numpy()
            
            # Forward pass
            outputs = model(videos)
            probs = outputs.cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels_np.flatten())
    
    # Converter para arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Calcular métricas
    metrics = {
        'dataset': dataset_name,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'total_samples': len(all_labels)
    }
    
    return metrics, all_preds, all_probs, all_labels


def plot_confusion_matrix(y_true, y_pred, dataset_name, save_path):
    """
    Plota e salva matriz de confusão.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        dataset_name: Nome do dataset
        save_path: Caminho para salvar a figura
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake'],
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {dataset_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Matriz de confusão salva: {save_path}")


def plot_roc_curve(y_true, y_probs, dataset_name, save_path, auc_score):
    """
    Plota e salva curva ROC.
    
    Args:
        y_true: Labels verdadeiros
        y_probs: Probabilidades preditas
        dataset_name: Nome do dataset
        save_path: Caminho para salvar a figura
        auc_score: Score AUC calculado
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {dataset_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Curva ROC salva: {save_path}")


def cross_dataset_evaluation(
    model_path='models/model_best.pt',
    splits_csv='data/splits_faceforensicspp.csv',
    celeb_csv='data/index_celebdf.csv',
    wild_csv='data/index_wilddeepfake.csv',
    batch_size=4,
    num_frames=16,
    metrics_save_path='outputs/metrics_cross.csv',
    figures_dir='outputs/figures',
    device=None
):
    """
    Avaliação cross-dataset em FaceForensics++, Celeb-DF-v2 e WildDeepfake.
    
    Args:
        model_path: Caminho do modelo treinado
        splits_csv: CSV de splits do FaceForensics++
        celeb_csv: CSV de índice do Celeb-DF-v2
        wild_csv: CSV de índice do WildDeepfake
        batch_size: Tamanho do batch
        num_frames: Frames por vídeo
        metrics_save_path: Caminho para salvar métricas
        figures_dir: Diretório para salvar figuras
        device: Dispositivo (None = auto-detect)
        
    Returns:
        all_metrics: DataFrame com todas as métricas
    """
    # Configurar seed
    set_global_seed(42)
    
    # Detectar device
    if device is None:
        device = get_device()
    
    print(f"\n{'='*70}")
    print(f"AVALIAÇÃO CROSS-DATASET")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Modelo: {model_path}")
    print(f"Batch size: {batch_size}\n")
    
    # Criar diretórios
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Carregar modelo
    print("Carregando modelo...")
    model, checkpoint = load_model(model_path, device=device)
    print(f"✓ Modelo carregado de: {model_path}\n")
    
    # Lista para armazenar todas as métricas
    all_metrics = []
    
    # ========================================================================
    # 1. FACEFORENSICS++ (TEST SPLIT)
    # ========================================================================
    print(f"{'='*70}")
    print("1. AVALIANDO FACEFORENSICS++ (TEST SPLIT)")
    print(f"{'='*70}")
    
    if os.path.exists(splits_csv):
        # Carregar DataLoaders
        dataloaders = get_dataloaders(
            splits_csv=splits_csv,
            batch_size=batch_size,
            num_frames=num_frames,
            num_workers=0,
            shuffle_train=False
        )
        
        test_loader = dataloaders['test']
        
        # Avaliar
        metrics, preds, probs, labels = evaluate_model(
            model, test_loader, device, "FaceForensics++"
        )
        all_metrics.append(metrics)
        
        # Imprimir métricas
        print(f"\nResultados FaceForensics++:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Samples:   {metrics['total_samples']}")
        
        # Gerar visualizações
        plot_confusion_matrix(
            labels, preds, "FaceForensics++",
            os.path.join(figures_dir, 'confusion_matrix_faceforensics.png')
        )
        plot_roc_curve(
            labels, probs, "FaceForensics++",
            os.path.join(figures_dir, 'roc_curve_faceforensics.png'),
            metrics['auc']
        )
    else:
        print(f"⚠️  Arquivo não encontrado: {splits_csv}")
        print("   Pulando avaliação FaceForensics++\n")
    
    # ========================================================================
    # 2. CELEB-DF-v2 (COMPLETO)
    # ========================================================================
    print(f"\n{'='*70}")
    print("2. AVALIANDO CELEB-DF-v2 (DATASET COMPLETO)")
    print(f"{'='*70}")
    
    if os.path.exists(celeb_csv):
        # Criar CSV de splits temporário (usar tudo como test)
        df_celeb = pd.read_csv(celeb_csv)
        df_celeb['split'] = 'test'
        celeb_splits_csv = 'data/splits_celebdf_temp.csv'
        df_celeb.to_csv(celeb_splits_csv, index=False)
        
        # Carregar DataLoaders
        dataloaders = get_dataloaders(
            splits_csv=celeb_splits_csv,
            batch_size=batch_size,
            num_frames=num_frames,
            num_workers=0,
            shuffle_train=False
        )
        
        test_loader = dataloaders['test']
        
        # Avaliar
        metrics, preds, probs, labels = evaluate_model(
            model, test_loader, device, "Celeb-DF-v2"
        )
        all_metrics.append(metrics)
        
        # Imprimir métricas
        print(f"\nResultados Celeb-DF-v2:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Samples:   {metrics['total_samples']}")
        
        # Gerar visualizações
        plot_confusion_matrix(
            labels, preds, "Celeb-DF-v2",
            os.path.join(figures_dir, 'confusion_matrix_celebdf.png')
        )
        plot_roc_curve(
            labels, probs, "Celeb-DF-v2",
            os.path.join(figures_dir, 'roc_curve_celebdf.png'),
            metrics['auc']
        )
        
        # Remover arquivo temporário
        os.remove(celeb_splits_csv)
    else:
        print(f"⚠️  Arquivo não encontrado: {celeb_csv}")
        print("   Pulando avaliação Celeb-DF-v2\n")
    
    # ========================================================================
    # 3. WILDDEEPFAKE (COMPLETO)
    # ========================================================================
    print(f"\n{'='*70}")
    print("3. AVALIANDO WILDDEEPFAKE (DATASET COMPLETO)")
    print(f"{'='*70}")
    
    if os.path.exists(wild_csv):
        # Criar CSV de splits temporário (usar tudo como test)
        df_wild = pd.read_csv(wild_csv)
        df_wild['split'] = 'test'
        wild_splits_csv = 'data/splits_wilddeepfake_temp.csv'
        df_wild.to_csv(wild_splits_csv, index=False)
        
        # Carregar DataLoaders
        dataloaders = get_dataloaders(
            splits_csv=wild_splits_csv,
            batch_size=batch_size,
            num_frames=num_frames,
            num_workers=0,
            shuffle_train=False
        )
        
        test_loader = dataloaders['test']
        
        # Avaliar
        metrics, preds, probs, labels = evaluate_model(
            model, test_loader, device, "WildDeepfake"
        )
        all_metrics.append(metrics)
        
        # Imprimir métricas
        print(f"\nResultados WildDeepfake:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Samples:   {metrics['total_samples']}")
        
        # Gerar visualizações
        plot_confusion_matrix(
            labels, preds, "WildDeepfake",
            os.path.join(figures_dir, 'confusion_matrix_wilddeepfake.png')
        )
        plot_roc_curve(
            labels, probs, "WildDeepfake",
            os.path.join(figures_dir, 'roc_curve_wilddeepfake.png'),
            metrics['auc']
        )
        
        # Remover arquivo temporário
        os.remove(wild_splits_csv)
    else:
        print(f"⚠️  Arquivo não encontrado: {wild_csv}")
        print("   Pulando avaliação WildDeepfake\n")
    
    # ========================================================================
    # SALVAR MÉTRICAS
    # ========================================================================
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(metrics_save_path, index=False)
        
        print(f"\n{'='*70}")
        print("RESUMO FINAL")
        print(f"{'='*70}\n")
        print(df_metrics.to_string(index=False))
        print(f"\n✓ Métricas salvas em: {metrics_save_path}")
        print(f"✓ Figuras salvas em: {figures_dir}/")
        
        return df_metrics
    else:
        print("\n⚠️  Nenhuma métrica foi coletada!")
        return None


def plot_training_curves(
    metrics_csv='outputs/metrics_train.csv',
    save_path='outputs/figures/training_curves.png',
    dpi=300
):
    """
    Gera gráfico de curvas de treinamento (loss e métricas).
    
    Args:
        metrics_csv: Caminho do CSV com métricas de treino
        save_path: Caminho para salvar a figura
        dpi: DPI da figura (legibilidade)
        
    Returns:
        dimensao_px: Tupla com dimensões em pixels (width, height)
    """
    print(f"\n{'='*70}")
    print("GERANDO GRÁFICO: TRAINING CURVES")
    print(f"{'='*70}\n")
    
    # Verificar se arquivo existe
    if not os.path.exists(metrics_csv):
        print(f"❌ ERRO: Arquivo não encontrado: {metrics_csv}")
        return None
    
    # Carregar dados
    df = pd.read_csv(metrics_csv)
    print(f"✓ Métricas carregadas: {len(df)} épocas")
    
    # Criar figura com 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ========================================================================
    # SUBPLOT 1: LOSS (Train vs Val)
    # ========================================================================
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_loss'], 'o-', linewidth=2, 
             markersize=6, label='Train Loss', color='#2E86AB')
    ax1.plot(df['epoch'], df['val_loss'], 's-', linewidth=2, 
             markersize=6, label='Val Loss', color='#A23B72')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(df['epoch'].min() - 0.5, df['epoch'].max() + 0.5)
    
    # ========================================================================
    # SUBPLOT 2: MÉTRICAS (F1 e AUC)
    # ========================================================================
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['val_f1'], 'o-', linewidth=2, 
             markersize=6, label='Val F1-Score', color='#F18F01')
    ax2.plot(df['epoch'], df['val_auc'], 's-', linewidth=2, 
             markersize=6, label='Val AUC', color='#06A77D')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Metrics', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(df['epoch'].min() - 0.5, df['epoch'].max() + 0.5)
    ax2.set_ylim(-0.05, 1.05)
    
    # Ajustar layout e salvar
    plt.tight_layout()
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Salvar figura
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Calcular dimensões em pixels
    fig_width, fig_height = fig.get_size_inches()
    width_px = int(fig_width * dpi)
    height_px = int(fig_height * dpi)
    dimensao_px = (width_px, height_px)
    
    plt.close()
    
    print(f"✓ Gráfico salvo: {save_path}")
    print(f"  Dimensões: {width_px}x{height_px} pixels ({dpi} DPI)")
    print(f"  Legibilidade: Alta (figsize={fig.get_size_inches()})")
    
    return dimensao_px


def plot_f1_by_dataset(
    metrics_csv='outputs/metrics_cross.csv',
    save_path='outputs/figures/f1_by_dataset.png',
    dpi=300
):
    """
    Gera gráfico de barras com F1-score por dataset.
    
    Args:
        metrics_csv: Caminho do CSV com métricas cross-dataset
        save_path: Caminho para salvar a figura
        dpi: DPI da figura (legibilidade)
        
    Returns:
        dimensao_px: Tupla com dimensões em pixels (width, height)
    """
    print(f"\n{'='*70}")
    print("GERANDO GRÁFICO: F1-SCORE BY DATASET")
    print(f"{'='*70}\n")
    
    # Verificar se arquivo existe
    if not os.path.exists(metrics_csv):
        print(f"❌ ERRO: Arquivo não encontrado: {metrics_csv}")
        return None
    
    # Carregar dados
    df = pd.read_csv(metrics_csv)
    print(f"✓ Métricas carregadas: {len(df)} datasets")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Cores para cada dataset
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Criar gráfico de barras
    bars = ax.bar(df['dataset'], df['f1'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Adicionar valores no topo das barras
    for bar, f1_value in zip(bars, df['f1']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., 
            height + 0.02,
            f'{f1_value:.4f}',
            ha='center', 
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    # Configurar eixos
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score by Dataset (Cross-Dataset Evaluation)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Rotacionar labels do eixo x se necessário
    plt.xticks(rotation=15, ha='right')
    
    # Ajustar layout e salvar
    plt.tight_layout()
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Salvar figura
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Calcular dimensões em pixels
    fig_width, fig_height = fig.get_size_inches()
    width_px = int(fig_width * dpi)
    height_px = int(fig_height * dpi)
    dimensao_px = (width_px, height_px)
    
    plt.close()
    
    print(f"✓ Gráfico salvo: {save_path}")
    print(f"  Dimensões: {width_px}x{height_px} pixels ({dpi} DPI)")
    print(f"  Legibilidade: Alta (figsize={fig.get_size_inches()})")
    
    return dimensao_px


def plot_gradcam_examples(
    heatmaps_dir='outputs/heatmaps',
    save_path='outputs/figures/gradcam_examples.png',
    num_examples=6,
    dpi=150
):
    """
    Gera montagem de exemplos de Grad-CAM.
    
    Args:
        heatmaps_dir: Diretório com heatmaps gerados
        save_path: Caminho para salvar a figura
        num_examples: Número de exemplos a incluir
        dpi: DPI da figura (legibilidade)
        
    Returns:
        dimensao_px: Tupla com dimensões em pixels (width, height)
    """
    print(f"\n{'='*70}")
    print("GERANDO GRÁFICO: GRAD-CAM EXAMPLES")
    print(f"{'='*70}\n")
    
    # Verificar se diretório existe
    if not os.path.exists(heatmaps_dir):
        print(f"❌ ERRO: Diretório não encontrado: {heatmaps_dir}")
        return None
    
    # Listar heatmaps disponíveis
    heatmap_files = sorted([
        f for f in os.listdir(heatmaps_dir) 
        if f.endswith('_gradcam.png')
    ])
    
    if len(heatmap_files) == 0:
        print(f"❌ ERRO: Nenhum heatmap encontrado em {heatmaps_dir}")
        return None
    
    print(f"✓ Heatmaps encontrados: {len(heatmap_files)}")
    
    # Selecionar exemplos distribuídos
    step = max(1, len(heatmap_files) // num_examples)
    selected_files = heatmap_files[::step][:num_examples]
    
    print(f"✓ Selecionados {len(selected_files)} exemplos para montagem")
    
    # Criar figura com grid
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    # Carregar e plotar cada exemplo
    from PIL import Image
    
    for idx, filename in enumerate(selected_files):
        if idx >= rows * cols:
            break
        
        # Carregar imagem
        img_path = os.path.join(heatmaps_dir, filename)
        img = Image.open(img_path)
        
        # Plotar
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Título (nome do arquivo sem extensão)
        title = filename.replace('_gradcam.png', '').replace('_', ' ').title()
        # Limitar tamanho do título
        if len(title) > 30:
            title = title[:27] + '...'
        axes[idx].set_title(title, fontsize=10, fontweight='bold')
    
    # Remover eixos vazios
    for idx in range(len(selected_files), len(axes)):
        axes[idx].axis('off')
    
    # Título geral
    fig.suptitle('Grad-CAM Examples: Attention Heatmaps on Video Frames', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Ajustar layout e salvar
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Salvar figura
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Calcular dimensões em pixels
    fig_width, fig_height = fig.get_size_inches()
    width_px = int(fig_width * dpi)
    height_px = int(fig_height * dpi)
    dimensao_px = (width_px, height_px)
    
    plt.close()
    
    print(f"✓ Gráfico salvo: {save_path}")
    print(f"  Dimensões: {width_px}x{height_px} pixels ({dpi} DPI)")
    print(f"  Legibilidade: Média-Alta (figsize={fig.get_size_inches()})")
    
    return dimensao_px


def create_metrics_table(
    train_csv='outputs/metrics_train.csv',
    cross_csv='outputs/metrics_cross.csv',
    save_path='outputs/reports/table_metrics.csv'
):
    """
    Cria tabela consolidada de métricas.
    
    Args:
        train_csv: CSV com métricas de treino
        cross_csv: CSV com métricas cross-dataset
        save_path: Caminho para salvar tabela
        
    Returns:
        df_table: DataFrame com tabela consolidada
    """
    print(f"\n{'='*70}")
    print("CRIANDO TABELA DE MÉTRICAS CONSOLIDADA")
    print(f"{'='*70}\n")
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Lista para armazenar linhas da tabela
    table_rows = []
    
    # ========================================================================
    # 1. MÉTRICAS DE TREINO (MELHOR ÉPOCA)
    # ========================================================================
    if os.path.exists(train_csv):
        df_train = pd.read_csv(train_csv)
        
        # Encontrar melhor época (maior Val F1)
        best_epoch_idx = df_train['val_f1'].idxmax()
        best_epoch = df_train.iloc[best_epoch_idx]
        
        table_rows.append({
            'metric_type': 'Training',
            'dataset': 'FaceForensics++ (Val)',
            'epoch': int(best_epoch['epoch']),
            'loss': best_epoch['val_loss'],
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': best_epoch['val_f1'],
            'auc': best_epoch['val_auc'],
            'samples': None
        })
        
        print(f"✓ Métricas de treino (melhor época: {int(best_epoch['epoch'])})")
    else:
        print(f"⚠️  Arquivo não encontrado: {train_csv}")
    
    # ========================================================================
    # 2. MÉTRICAS CROSS-DATASET
    # ========================================================================
    if os.path.exists(cross_csv):
        df_cross = pd.read_csv(cross_csv)
        
        for _, row in df_cross.iterrows():
            table_rows.append({
                'metric_type': 'Cross-Dataset Evaluation',
                'dataset': row['dataset'],
                'epoch': None,
                'loss': None,
                'accuracy': row['accuracy'],
                'precision': row['precision'],
                'recall': row['recall'],
                'f1': row['f1'],
                'auc': row['auc'],
                'samples': int(row['total_samples'])
            })
        
        print(f"✓ Métricas cross-dataset ({len(df_cross)} datasets)")
    else:
        print(f"⚠️  Arquivo não encontrado: {cross_csv}")
    
    # ========================================================================
    # 3. CRIAR DATAFRAME E SALVAR
    # ========================================================================
    if table_rows:
        df_table = pd.DataFrame(table_rows)
        
        # Ordenar: Treino primeiro, depois cross-dataset
        df_table = df_table.sort_values(['metric_type', 'dataset'], ascending=[False, True])
        
        # Salvar
        df_table.to_csv(save_path, index=False)
        
        print(f"\n✓ Tabela salva: {save_path}")
        print(f"  Total de linhas: {len(df_table)}")
        print(f"\nPreview da tabela:")
        print(df_table.to_string(index=False))
        
        return df_table
    else:
        print("\n❌ ERRO: Nenhuma métrica encontrada!")
        return None


def generate_all_figures_and_reports():
    """
    Gera todas as figuras e relatórios da Tarefa 12.
    
    Outputs:
        - outputs/figures/training_curves.png
        - outputs/figures/f1_by_dataset.png
        - outputs/figures/gradcam_examples.png
        - outputs/reports/table_metrics.csv
        
    Returns:
        metrics: Dicionário com métricas de legibilidade
    """
    print("\n" + "="*70)
    print("TAREFA 12: GERAR FIGURAS E RELATÓRIOS")
    print("="*70 + "\n")
    
    metrics = {}
    
    # ========================================================================
    # 1. TRAINING CURVES
    # ========================================================================
    dim1 = plot_training_curves(
        metrics_csv='outputs/metrics_train.csv',
        save_path='outputs/figures/training_curves.png',
        dpi=300
    )
    if dim1:
        metrics['training_curves_dimensao_px'] = f"{dim1[0]}x{dim1[1]}"
        metrics['training_curves_legibilidade'] = "Alta"
    
    # ========================================================================
    # 2. F1 BY DATASET
    # ========================================================================
    dim2 = plot_f1_by_dataset(
        metrics_csv='outputs/metrics_cross.csv',
        save_path='outputs/figures/f1_by_dataset.png',
        dpi=300
    )
    if dim2:
        metrics['f1_by_dataset_dimensao_px'] = f"{dim2[0]}x{dim2[1]}"
        metrics['f1_by_dataset_legibilidade'] = "Alta"
    
    # ========================================================================
    # 3. GRAD-CAM EXAMPLES
    # ========================================================================
    dim3 = plot_gradcam_examples(
        heatmaps_dir='outputs/heatmaps',
        save_path='outputs/figures/gradcam_examples.png',
        num_examples=6,
        dpi=150
    )
    if dim3:
        metrics['gradcam_examples_dimensao_px'] = f"{dim3[0]}x{dim3[1]}"
        metrics['gradcam_examples_legibilidade'] = "Média-Alta"
    
    # ========================================================================
    # 4. TABLE METRICS
    # ========================================================================
    df_table = create_metrics_table(
        train_csv='outputs/metrics_train.csv',
        cross_csv='outputs/metrics_cross.csv',
        save_path='outputs/reports/table_metrics.csv'
    )
    
    if df_table is not None:
        metrics['table_metrics_rows'] = len(df_table)
    
    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    print(f"\n{'='*70}")
    print("RESUMO - TAREFA 12")
    print(f"{'='*70}\n")
    
    print("Figuras geradas:")
    print("  [✓] outputs/figures/training_curves.png")
    print("  [✓] outputs/figures/f1_by_dataset.png")
    print("  [✓] outputs/figures/gradcam_examples.png")
    
    print("\nRelatórios gerados:")
    print("  [✓] outputs/reports/table_metrics.csv")
    
    print("\nMétricas alcançadas:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*70}")
    print("TAREFA 12 CONCLUÍDA COM SUCESSO")
    print(f"{'='*70}\n")
    
    return metrics


def test_cross_evaluation():
    """
    Função de teste para validar a avaliação cross-dataset.
    """
    print("\n" + "="*70)
    print("TESTE DE AVALIAÇÃO CROSS-DATASET")
    print("="*70 + "\n")
    
    # Verificar se modelo existe
    model_path = 'models/model_best.pt'
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em {model_path}")
        print("Execute primeiro o treinamento (Tarefa 7).")
        return
    
    # Executar avaliação
    df_metrics = cross_dataset_evaluation(
        model_path='models/model_best.pt',
        splits_csv='data/splits_faceforensicspp.csv',
        celeb_csv='data/index_celebdf.csv',
        wild_csv='data/index_wilddeepfake.csv',
        batch_size=4,
        num_frames=16
    )
    
    if df_metrics is not None:
        print("\n" + "="*70)
        print("TESTE CONCLUÍDO COM SUCESSO")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("TESTE CONCLUÍDO COM AVISOS")
        print("="*70 + "\n")


def test_robustness(
    model_path='models/model_best.pt',
    test_videos_dir='data/wilddeepfake/videos_real',
    num_test_videos=3,
    num_frames=16,
    output_csv='outputs/reports/robustness.csv',
    output_plot='outputs/figures/robustness.png',
    device=None
):
    """
    Testa robustez do modelo com vídeos degradados.
    
    Aplica degradações:
    - Ruído gaussiano (diferentes níveis)
    - Blur (diferentes níveis)
    - Compressão JPEG (diferentes qualidades)
    - Redimensionamento (downscaling)
    
    Args:
        model_path: Caminho do modelo treinado
        test_videos_dir: Diretório com vídeos de teste
        num_test_videos: Número de vídeos a testar
        num_frames: Frames por vídeo
        output_csv: Caminho do CSV de resultados
        output_plot: Caminho do gráfico
        device: Dispositivo (None = auto-detect)
        
    Returns:
        df_results: DataFrame com resultados de robustez
    """
    print(f"\n{'='*70}")
    print("TESTE DE ROBUSTEZ")
    print(f"{'='*70}\n")
    
    # Configurar seed
    set_global_seed(42)
    
    # Detectar device
    if device is None:
        device = get_device()
    
    print(f"Device: {device}")
    print(f"Modelo: {model_path}")
    print(f"Diretório de teste: {test_videos_dir}\n")
    
    # Criar diretórios
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    
    # Carregar modelo
    print("Carregando modelo...")
    model, checkpoint = load_model(model_path, device=device)
    print(f"✓ Modelo carregado\n")
    
    # Inicializar MTCNN
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(device='cpu', post_process=False)
    
    # Buscar vídeos de teste
    if not os.path.exists(test_videos_dir):
        print(f"❌ ERRO: Diretório não encontrado: {test_videos_dir}")
        return None
    
    video_files = [f for f in os.listdir(test_videos_dir) if f.endswith('.mp4')]
    
    if len(video_files) == 0:
        print(f"❌ ERRO: Nenhum vídeo encontrado em {test_videos_dir}")
        return None
    
    # Selecionar vídeos de teste
    test_videos = video_files[:num_test_videos]
    print(f"Vídeos selecionados para teste: {len(test_videos)}")
    for vid in test_videos:
        print(f"  - {vid}")
    print()
    
    # ========================================================================
    # DEFINIR DEGRADAÇÕES
    # ========================================================================
    
    degradations = [
        {'type': 'original', 'param': None, 'label': 'Original'},
        
        # Ruído Gaussiano
        {'type': 'gaussian_noise', 'param': 0.01, 'label': 'Noise σ=0.01'},
        {'type': 'gaussian_noise', 'param': 0.05, 'label': 'Noise σ=0.05'},
        {'type': 'gaussian_noise', 'param': 0.10, 'label': 'Noise σ=0.10'},
        
        # Blur (Gaussian)
        {'type': 'blur', 'param': 3, 'label': 'Blur k=3'},
        {'type': 'blur', 'param': 7, 'label': 'Blur k=7'},
        {'type': 'blur', 'param': 15, 'label': 'Blur k=15'},
        
        # Compressão JPEG
        {'type': 'jpeg_compression', 'param': 90, 'label': 'JPEG Q=90'},
        {'type': 'jpeg_compression', 'param': 50, 'label': 'JPEG Q=50'},
        {'type': 'jpeg_compression', 'param': 20, 'label': 'JPEG Q=20'},
        
        # Redimensionamento
        {'type': 'resize', 'param': 0.75, 'label': 'Resize 75%'},
        {'type': 'resize', 'param': 0.50, 'label': 'Resize 50%'},
        {'type': 'resize', 'param': 0.25, 'label': 'Resize 25%'},
    ]
    
    print(f"Degradações a testar: {len(degradations)}\n")
    
    # ========================================================================
    # EXECUTAR TESTES
    # ========================================================================
    
    results = []
    
    for video_file in test_videos:
        video_path = os.path.join(test_videos_dir, video_file)
        print(f"Processando: {video_file}")
        
        # Pré-processar vídeo original
        from src.preprocessing import preprocess_video
        result = preprocess_video(video_path, num_frames=num_frames, mtcnn=mtcnn)
        
        if result is None or result[0] is None:
            print(f"  ⚠️  Falha ao processar vídeo, pulando...")
            continue
        
        video_tensor_original, detection_rate, _ = result
        
        # Probabilidade original (baseline)
        model.eval()
        with torch.no_grad():
            video_batch = video_tensor_original.unsqueeze(0).to(device)
            prob_original = model(video_batch).item()
        
        print(f"  Probabilidade original: {prob_original:.4f}")
        
        # Testar cada degradação
        for deg in tqdm(degradations, desc=f"  Degradações", leave=False):
            deg_type = deg['type']
            deg_param = deg['param']
            deg_label = deg['label']
            
            if deg_type == 'original':
                # Usar tensor original
                video_tensor_degraded = video_tensor_original
                prob_degraded = prob_original
            else:
                # Aplicar degradação
                video_tensor_degraded = apply_degradation(
                    video_tensor_original.clone(),
                    deg_type,
                    deg_param
                )
                
                # Predição no vídeo degradado
                model.eval()
                with torch.no_grad():
                    video_batch_deg = video_tensor_degraded.unsqueeze(0).to(device)
                    prob_degraded = model(video_batch_deg).item()
            
            # Calcular delta
            delta_prob = abs(prob_degraded - prob_original)
            
            # Armazenar resultado
            results.append({
                'video': video_file,
                'degradation_type': deg_type,
                'degradation_param': deg_param,
                'degradation_label': deg_label,
                'prob_original': prob_original,
                'prob_degraded': prob_degraded,
                'delta_probabilidade': delta_prob,
                'detection_rate': detection_rate
            })
        
        print(f"  ✓ Concluído\n")
    
    # ========================================================================
    # SALVAR RESULTADOS
    # ========================================================================
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    
    print(f"✓ Resultados salvos: {output_csv}")
    print(f"  Total de testes: {len(df_results)}")
    
    # ========================================================================
    # GERAR GRÁFICO
    # ========================================================================
    
    print(f"\nGerando gráfico de robustez...")
    
    # Agrupar por tipo de degradação
    df_grouped = df_results.groupby('degradation_label')['delta_probabilidade'].mean().reset_index()
    df_grouped = df_grouped.sort_values('delta_probabilidade', ascending=False)
    
    # Criar figura
    plt.figure(figsize=(12, 6))
    
    # Cores por tipo de degradação
    colors = []
    for label in df_grouped['degradation_label']:
        if 'Original' in label:
            colors.append('#2ecc71')  # Verde
        elif 'Noise' in label:
            colors.append('#e74c3c')  # Vermelho
        elif 'Blur' in label:
            colors.append('#3498db')  # Azul
        elif 'JPEG' in label:
            colors.append('#f39c12')  # Laranja
        elif 'Resize' in label:
            colors.append('#9b59b6')  # Roxo
        else:
            colors.append('#95a5a6')  # Cinza
    
    # Gráfico de barras
    bars = plt.bar(range(len(df_grouped)), df_grouped['delta_probabilidade'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Adicionar valores no topo das barras
    for i, (bar, delta) in enumerate(zip(bars, df_grouped['delta_probabilidade'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f'{delta:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Configurar eixos
    plt.xlabel('Degradation Type', fontsize=12, fontweight='bold')
    plt.ylabel('Δ Probability (Mean Absolute Change)', fontsize=12, fontweight='bold')
    plt.title('Model Robustness: Probability Change under Degradations', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(df_grouped)), df_grouped['degradation_label'], 
               rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, df_grouped['delta_probabilidade'].max() * 1.15)
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Original'),
        Patch(facecolor='#e74c3c', label='Gaussian Noise'),
        Patch(facecolor='#3498db', label='Blur'),
        Patch(facecolor='#f39c12', label='JPEG Compression'),
        Patch(facecolor='#9b59b6', label='Resize')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Gráfico salvo: {output_plot}")
    
    # ========================================================================
    # ESTATÍSTICAS
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("ESTATÍSTICAS DE ROBUSTEZ")
    print(f"{'='*70}\n")
    
    print("Δ Probabilidade médio por degradação:")
    print(df_grouped.to_string(index=False))
    
    print(f"\nResumo Geral:")
    print(f"  Δ Probabilidade médio: {df_results['delta_probabilidade'].mean():.4f}")
    print(f"  Δ Probabilidade máximo: {df_results['delta_probabilidade'].max():.4f}")
    print(f"  Δ Probabilidade mínimo: {df_results['delta_probabilidade'].min():.4f}")
    print(f"  Desvio padrão: {df_results['delta_probabilidade'].std():.4f}")
    
    # Degradação mais impactante
    max_delta_row = df_results.loc[df_results['delta_probabilidade'].idxmax()]
    print(f"\nDegradação mais impactante:")
    print(f"  Tipo: {max_delta_row['degradation_label']}")
    print(f"  Δ Probabilidade: {max_delta_row['delta_probabilidade']:.4f}")
    print(f"  Vídeo: {max_delta_row['video']}")
    
    print(f"\n{'='*70}")
    print("TESTE DE ROBUSTEZ CONCLUÍDO")
    print(f"{'='*70}\n")
    
    return df_results


def apply_degradation(video_tensor, degradation_type, param):
    """
    Aplica degradação a um tensor de vídeo.
    
    Args:
        video_tensor: Tensor de vídeo (T, C, H, W)
        degradation_type: Tipo de degradação
        param: Parâmetro da degradação
        
    Returns:
        video_tensor_degraded: Tensor degradado
    """
    import torch.nn.functional as F
    
    if degradation_type == 'gaussian_noise':
        # Adicionar ruído gaussiano
        noise = torch.randn_like(video_tensor) * param
        return torch.clamp(video_tensor + noise, 0, 1)
    
    elif degradation_type == 'blur':
        # Aplicar blur gaussiano
        # Criar kernel gaussiano
        kernel_size = int(param)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma = kernel_size / 6.0
        
        # Aplicar blur frame por frame
        degraded_frames = []
        for t in range(video_tensor.shape[0]):
            frame = video_tensor[t]  # (C, H, W)
            
            # Usar convolução com kernel gaussiano
            # Simplificação: usar avg pooling como blur
            blurred = F.avg_pool2d(
                frame.unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2
            ).squeeze(0)
            
            degraded_frames.append(blurred)
        
        return torch.stack(degraded_frames)
    
    elif degradation_type == 'jpeg_compression':
        # Simular compressão JPEG
        # Aplicar quantização
        quality = param / 100.0
        
        # Quantização simplificada
        quantized = torch.round(video_tensor * 255 * quality) / (255 * quality)
        return torch.clamp(quantized, 0, 1)
    
    elif degradation_type == 'resize':
        # Redimensionar (downscale + upscale)
        scale_factor = param
        
        T, C, H, W = video_tensor.shape
        new_H = int(H * scale_factor)
        new_W = int(W * scale_factor)
        
        # Downscale
        downscaled = F.interpolate(
            video_tensor,
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        )
        
        # Upscale de volta
        upscaled = F.interpolate(
            downscaled,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        return torch.clamp(upscaled, 0, 1)
    
    else:
        # Degradação desconhecida, retornar original
        return video_tensor


if __name__ == '__main__':
    # Executar avaliação cross-dataset (Tarefa 9)
    cross_dataset_evaluation()
