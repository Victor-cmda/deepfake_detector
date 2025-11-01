#!/usr/bin/env python
"""
Teste rápido para verificar se as correções funcionaram.
Executa apenas 1 época para validar que loss está OK.
"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

from src.train import train_model


def main():
    """Teste rápido - 1 época."""
    
    print("\n" + "="*70)
    print("TESTE RÁPIDO - VERIFICAÇÃO DAS CORREÇÕES")
    print("="*70 + "\n")
    
    config = {
        'splits_csv': 'data/splits_faceforensicspp.csv',
        'batch_size': 8,
        'num_frames': 16,
        'num_epochs': 1,  # ← APENAS 1 ÉPOCA PARA TESTE
        'learning_rate': 1e-4,
        'patience': 5,
        'num_workers': 0
    }
    
    print("Configuração de TESTE:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    print("Executando 1 época de teste...")
    print("Checklist de sucesso:")
    print("  [ ] Train Loss < 0.70")
    print("  [ ] Val Loss < 0.65")
    print("  [ ] Val AUC > 0.55")
    print("  [ ] Loss diminui durante época")
    print()
    
    model, history = train_model(**config)
    
    # Analisar resultados
    print("\n" + "="*70)
    print("RESULTADOS DO TESTE")
    print("="*70 + "\n")
    
    train_loss = history['train_loss'][0]
    val_loss = history['val_loss'][0]
    val_auc = history['val_auc'][0]
    val_f1 = history['val_f1'][0]
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val AUC:    {val_auc:.4f}")
    print(f"Val F1:     {val_f1:.4f}\n")
    
    # Verificar se passou nos critérios
    passed = True
    
    if train_loss < 0.70:
        print("✅ Train Loss < 0.70: PASSOU")
    else:
        print(f"❌ Train Loss < 0.70: FALHOU (atual: {train_loss:.4f})")
        passed = False
    
    if val_loss < 0.65:
        print("✅ Val Loss < 0.65: PASSOU")
    else:
        print(f"❌ Val Loss < 0.65: FALHOU (atual: {val_loss:.4f})")
        passed = False
    
    if val_auc > 0.55:
        print("✅ Val AUC > 0.55: PASSOU")
    else:
        print(f"❌ Val AUC > 0.55: FALHOU (atual: {val_auc:.4f})")
        passed = False
    
    print("\n" + "="*70)
    if passed:
        print("✅ TESTE PASSOU! Correções funcionaram.")
        print("\nPróximo passo:")
        print("  .venv-1\\Scripts\\python.exe train_full.py")
    else:
        print("❌ TESTE FALHOU! Verificar configuração.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
