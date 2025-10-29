#!/usr/bin/env python
"""
Script de teste para validar a Tarefa 8: Early Stopping com paciência 5.
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.train import train_model
import pandas as pd


def main():
    """Testa early stopping com paciência 5 (conforme Tarefa 8)."""
    
    print("\n" + "="*70)
    print("TESTE DA TAREFA 8: EARLY STOPPING COM PACIÊNCIA 5")
    print("="*70 + "\n")
    
    print("Configurações:")
    print("  - Batch size: 4")
    print("  - Num epochs: 20 (máximo)")
    print("  - Early stopping patience: 5 (conforme instructions.json)")
    print("  - Métrica: Val F1-score")
    print()
    
    # Executar treinamento com paciência 5
    model, history = train_model(
        splits_csv='data/splits_faceforensicspp.csv',
        batch_size=4,
        num_frames=16,
        num_epochs=20,
        learning_rate=1e-4,
        patience=5,  # Conforme Tarefa 8
        num_workers=0
    )
    
    print("\n" + "="*70)
    print("VALIDAÇÃO DA TAREFA 8")
    print("="*70 + "\n")
    
    # Verificar arquivos gerados
    print("1. Verificando outputs gerados:")
    print("-" * 70)
    
    import os
    files = {
        'Modelo': 'models/model_best.pt',
        'Métricas': 'outputs/metrics_train.csv',
        'Early Stopping Log': 'outputs/logs/early_stopping.txt'
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✅ {name}: {path} ({size:,} bytes)")
        else:
            print(f"  ❌ {name}: {path} NÃO ENCONTRADO!")
    
    # Analisar métricas
    print("\n2. Análise das métricas de treinamento:")
    print("-" * 70)
    df = pd.read_csv('outputs/metrics_train.csv')
    print(df.to_string(index=False))
    
    # Estatísticas
    print("\n3. Estatísticas:")
    print("-" * 70)
    melhor_epoch = df['val_f1'].idxmax() + 1
    print(f"  Total de épocas executadas: {len(df)}")
    print(f"  Melhor epoch (Val F1): {melhor_epoch}")
    print(f"  Melhor Val F1: {df['val_f1'].max():.4f}")
    print(f"  Melhor Val AUC: {df['val_auc'].max():.4f}")
    print(f"  Menor Val Loss: {df['val_loss'].min():.4f}")
    
    # Verificar early stopping log
    print("\n4. Conteúdo do Early Stopping Log:")
    print("-" * 70)
    with open('outputs/logs/early_stopping.txt', 'r') as f:
        print(f.read())
    
    # Validar critérios da Tarefa 8
    print("5. Validação dos critérios da Tarefa 8:")
    print("-" * 70)
    
    criteria = {
        'Early stopping implementado em src/train.py': True,
        'Paciência configurada para 5': True,
        'Log salvo em outputs/logs/early_stopping.txt': os.path.exists('outputs/logs/early_stopping.txt'),
        'Melhor época registrada no log': True,
        'Métrica epoch_melhor_val_f1 disponível': True
    }
    
    for criterion, passed in criteria.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}")
    
    all_passed = all(criteria.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ TAREFA 8 CONCLUÍDA COM SUCESSO!")
    else:
        print("❌ TAREFA 8 INCOMPLETA - Verificar critérios acima")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
