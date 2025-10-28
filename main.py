"""
Script principal do projeto Deepfake Detector.
Orquestra o pipeline completo de treino, avaliação e interface.
"""

import sys
import argparse
from src.utils import set_global_seed, get_device


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Deepfake Detector')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'interface'], 
                       required=True, help='Modo de execução')
    parser.add_argument('--seed', type=int, default=42, help='Seed global')
    
    args = parser.parse_args()
    
    # Configurar seed
    set_global_seed(args.seed)
    device = get_device()
    
    if args.mode == 'train':
        print("\n=== MODO: TREINAMENTO ===")
        # Será implementado na tarefa 7
        
    elif args.mode == 'eval':
        print("\n=== MODO: AVALIAÇÃO ===")
        # Será implementado na tarefa 9
        
    elif args.mode == 'interface':
        print("\n=== MODO: INTERFACE ===")
        # Será implementado na tarefa 11
    
    print("\n✓ Execução concluída!")


if __name__ == '__main__':
    main()
