#!/usr/bin/env python
"""
Script de teste para treinamento completo do modelo DeepfakeDetector.
Demonstra o funcionamento do treinamento com early stopping.
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.train import train_model
import pandas as pd


def main():
    """Executa treinamento completo do modelo."""
    
    print("\n" + "="*70)
    print("TREINAMENTO COMPLETO DO MODELO DEEPFAKE DETECTOR")
    print("="*70 + "\n")
    
    # Configurações de treinamento otimizadas
    config = {
        'splits_csv': 'data/splits_faceforensicspp.csv',
        'batch_size': 8,  # Aumentado de 4 para 8 (Mixed Precision permite mais)
        'num_frames': 16,
        'num_epochs': 20,  # Máximo de épocas
        'learning_rate': 1e-4,
        'patience': 5,  # Early stopping baseado em Val AUC
        'num_workers': 0
    }
    
    print("Configurações:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Executar treinamento
    model, history = train_model(**config)
    
    # Análise dos resultados
    print("\n" + "="*70)
    print("ANÁLISE DOS RESULTADOS")
    print("="*70 + "\n")
    
    # Carregar métricas
    df = pd.read_csv('outputs/metrics_train.csv')
    
    print("Resumo das métricas:")
    print("-" * 70)
    print(df.to_string(index=False))
    print()
    
    # Estatísticas
    print("\nEstatísticas:")
    print(f"  Total de épocas executadas: {len(df)}")
    print(f"  Melhor Val F1: {df['val_f1'].max():.4f} (época {df['val_f1'].idxmax() + 1})")
    print(f"  Melhor Val AUC: {df['val_auc'].max():.4f} (época {df['val_auc'].idxmax() + 1})")
    print(f"  Menor Val Loss: {df['val_loss'].min():.4f} (época {df['val_loss'].idxmin() + 1})")
    print(f"  Train Loss final: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Val Loss final: {df['val_loss'].iloc[-1]:.4f}")
    print(f"  Learning Rate final: {df['learning_rate'].iloc[-1]:.2e}")
    
    # Verificar early stopping
    print("\nEarly Stopping:")
    with open('outputs/logs/early_stopping.txt', 'r') as f:
        content = f.read()
        print(content)
    
    print("="*70)
    print("TREINAMENTO FINALIZADO COM SUCESSO!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
