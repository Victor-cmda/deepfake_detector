#!/usr/bin/env python
"""
Teste rápido do treinamento - Executa 2 épocas para validar o pipeline.
"""

import sys
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.train import train_model
import pandas as pd


def main():
    """Executa teste rápido do treinamento."""
    
    print("\n" + "="*70)
    print("TESTE RÁPIDO DO PIPELINE DE TREINAMENTO")
    print("="*70 + "\n")
    
    # Configurações de teste (poucas épocas)
    config = {
        'splits_csv': 'data/splits_faceforensicspp.csv',
        'batch_size': 2,          # Batch pequeno
        'num_frames': 8,          # Menos frames
        'num_epochs': 2,          # Apenas 2 épocas
        'learning_rate': 1e-4,
        'patience': 10,           # Desabilitar early stopping
        'num_workers': 0
    }
    
    print("Configurações de teste:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n⏱️  Iniciando treinamento de teste...\n")
    
    # Executar treinamento
    try:
        model, history = train_model(**config)
        
        print("\n✅ TESTE CONCLUÍDO COM SUCESSO!")
        print("\nResumo:")
        if 'outputs/metrics_train.csv' in str(Path('outputs/metrics_train.csv')):
            df = pd.read_csv('outputs/metrics_train.csv')
            print(f"  Épocas executadas: {len(df)}")
            if len(df) > 0:
                print(f"  Val F1 final: {df['val_f1'].iloc[-1]:.4f}")
                print(f"  Val Loss final: {df['val_loss'].iloc[-1]:.4f}")
        
        print("\n" + "="*70)
        print("Pipeline validado! O sistema está funcionando corretamente.")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO NO TESTE: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
