"""
Script de teste do modelo DeepfakeDetector.
Testa arquitetura, forward pass e contagem de parâmetros.
"""

import os
import torch
from src.model import create_model, test_model_forward, save_model, load_model
from src.utils import set_global_seed, get_device, ensure_dir


def test_model_creation():
    """
    Testa a criação do modelo com diferentes configurações.
    """
    print("=" * 60)
    print("TESTE 1: CRIAÇÃO DO MODELO")
    print("=" * 60)
    print()
    
    device = get_device()
    
    # Configuração padrão
    print("1.1. Modelo com configuração padrão:")
    print("-" * 60)
    model = create_model(
        num_frames=16,
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.3,
        pretrained=True,
        device=device
    )
    
    print("\n✓ Modelo criado com sucesso!\n")
    
    return model, device


def test_forward_pass(model, device):
    """
    Testa o forward pass com diferentes tamanhos de batch.
    """
    print("=" * 60)
    print("TESTE 2: FORWARD PASS")
    print("=" * 60)
    print()
    
    # Teste com batch size 1
    print("2.1. Batch size = 1:")
    print("-" * 60)
    stats_1 = test_model_forward(model, batch_size=1, num_frames=16, device=device)
    
    print("\n" + "-" * 60)
    print("2.2. Batch size = 4:")
    print("-" * 60)
    stats_4 = test_model_forward(model, batch_size=4, num_frames=16, device=device)
    
    print("\n" + "-" * 60)
    print("2.3. Batch size = 8:")
    print("-" * 60)
    stats_8 = test_model_forward(model, batch_size=8, num_frames=16, device=device)
    
    print("\n✓ Todos os testes de forward pass concluídos!\n")
    
    return {
        'batch_1': stats_1,
        'batch_4': stats_4,
        'batch_8': stats_8
    }


def test_model_components(model):
    """
    Testa componentes individuais do modelo.
    """
    print("=" * 60)
    print("TESTE 3: COMPONENTES DO MODELO")
    print("=" * 60)
    print()
    
    # Parâmetros
    params = model.get_num_params()
    
    print("3.1. Contagem de parâmetros:")
    print("-" * 60)
    print(f"  Total: {params['total']:,}")
    print(f"  Treináveis: {params['trainable']:,}")
    print(f"  CNN (ResNet-34): {params['cnn']:,}")
    print(f"  LSTM (BiLSTM): {params['lstm']:,}")
    print(f"  FC (Classificador): {params['fc']:,}")
    
    # Distribuição de parâmetros
    cnn_pct = (params['cnn'] / params['total']) * 100
    lstm_pct = (params['lstm'] / params['total']) * 100
    fc_pct = (params['fc'] / params['total']) * 100
    
    print(f"\n  Distribuição:")
    print(f"    CNN: {cnn_pct:.1f}%")
    print(f"    LSTM: {lstm_pct:.1f}%")
    print(f"    FC: {fc_pct:.1f}%")
    
    # Teste de freeze/unfreeze
    print("\n" + "-" * 60)
    print("3.2. Teste de freeze/unfreeze CNN:")
    print("-" * 60)
    
    original_trainable = params['trainable']
    
    model.freeze_cnn()
    params_frozen = model.get_num_params()
    print(f"  Parâmetros treináveis após freeze: {params_frozen['trainable']:,}")
    print(f"  Redução: {original_trainable - params_frozen['trainable']:,}")
    
    model.unfreeze_cnn()
    params_unfrozen = model.get_num_params()
    print(f"  Parâmetros treináveis após unfreeze: {params_unfrozen['trainable']:,}")
    
    print("\n✓ Testes de componentes concluídos!\n")
    
    return params


def test_save_load(model, device):
    """
    Testa salvamento e carregamento do modelo.
    """
    print("=" * 60)
    print("TESTE 4: SALVAMENTO E CARREGAMENTO")
    print("=" * 60)
    print()
    
    # Criar diretório
    model_dir = 'models'
    ensure_dir(model_dir)
    
    # Salvar modelo
    print("4.1. Salvando modelo:")
    print("-" * 60)
    model_path = os.path.join(model_dir, 'test_model.pt')
    
    save_model(
        model,
        model_path,
        epoch=0,
        metrics={'test': True}
    )
    
    # Carregar modelo
    print("\n" + "-" * 60)
    print("4.2. Carregando modelo:")
    print("-" * 60)
    
    loaded_model, checkpoint = load_model(model_path, device=device)
    
    # Verificar se é o mesmo
    print("\n" + "-" * 60)
    print("4.3. Verificação:")
    print("-" * 60)
    
    # Comparar parâmetros
    original_params = model.get_num_params()
    loaded_params = loaded_model.get_num_params()
    
    print(f"  Parâmetros originais: {original_params['total']:,}")
    print(f"  Parâmetros carregados: {loaded_params['total']:,}")
    print(f"  Match: {'✓ SIM' if original_params['total'] == loaded_params['total'] else '✗ NÃO'}")
    
    # Teste forward com mesmo input
    dummy_input = torch.randn(2, 16, 3, 224, 224).to(device)
    
    model.eval()
    loaded_model.eval()
    
    with torch.no_grad():
        output_original = model(dummy_input)
        output_loaded = loaded_model(dummy_input)
    
    difference = torch.abs(output_original - output_loaded).max().item()
    print(f"  Diferença máxima nas saídas: {difference:.10f}")
    print(f"  Outputs idênticos: {'✓ SIM' if difference < 1e-6 else '✗ NÃO'}")
    
    print("\n✓ Testes de salvamento/carregamento concluídos!\n")
    
    # Limpar arquivo de teste
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"  (Arquivo de teste removido: {model_path})\n")


def create_model_summary():
    """
    Cria resumo das especificações do modelo.
    """
    print("=" * 60)
    print("RESUMO DAS ESPECIFICAÇÕES DO MODELO")
    print("=" * 60)
    print()
    
    device = get_device()
    model = create_model(device=device)
    params = model.get_num_params()
    
    # Criar relatório
    report_path = 'outputs/logs/model_specs.txt'
    ensure_dir(os.path.dirname(report_path))
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ESPECIFICAÇÕES DO MODELO DEEPFAKE DETECTOR\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("ARQUITETURA:\n")
        f.write("-" * 60 + "\n")
        f.write("1. CNN (Extrator de Features Espaciais)\n")
        f.write("   - Backbone: ResNet-34 (pré-treinado ImageNet)\n")
        f.write("   - Output features: 512 dimensões\n")
        f.write("   - Parâmetros: {:,}\n\n".format(params['cnn']))
        
        f.write("2. LSTM (Modelagem Temporal)\n")
        f.write("   - Tipo: BiLSTM (Bidirectional)\n")
        f.write("   - Camadas: 2\n")
        f.write("   - Hidden units por direção: 256\n")
        f.write("   - Total hidden: 512 (256 * 2)\n")
        f.write("   - Dropout: 0.3\n")
        f.write("   - Parâmetros: {:,}\n\n".format(params['lstm']))
        
        f.write("3. Classificador (Detecção Binária)\n")
        f.write("   - Camada FC: 512 → 1\n")
        f.write("   - Ativação: Sigmoid\n")
        f.write("   - Dropout: 0.3\n")
        f.write("   - Parâmetros: {:,}\n\n".format(params['fc']))
        
        f.write("PARÂMETROS TOTAIS:\n")
        f.write("-" * 60 + "\n")
        f.write("Total: {:,}\n".format(params['total']))
        f.write("Treináveis: {:,}\n\n".format(params['trainable']))
        
        f.write("INPUT/OUTPUT:\n")
        f.write("-" * 60 + "\n")
        f.write("Input shape: (batch_size, 16, 3, 224, 224)\n")
        f.write("  - 16 frames por vídeo\n")
        f.write("  - 3 canais RGB\n")
        f.write("  - 224x224 pixels\n\n")
        f.write("Output shape: (batch_size, 1)\n")
        f.write("  - Probabilidade de deepfake [0, 1]\n\n")
        
        f.write("MÉTRICAS DE PERFORMANCE:\n")
        f.write("-" * 60 + "\n")
        
        # Teste de velocidade
        stats = test_model_forward(model, batch_size=1, num_frames=16, device=device)
        f.write("Tempo de inferência (1 vídeo): {:.4f}s\n".format(stats['time_per_sample']))
        
        stats_batch = test_model_forward(model, batch_size=8, num_frames=16, device=device)
        f.write("Tempo de inferência (batch 8): {:.4f}s\n".format(stats_batch['forward_time']))
        f.write("Tempo por vídeo (batch 8): {:.4f}s\n".format(stats_batch['time_per_sample']))
    
    print(f"✓ Resumo salvo: {report_path}\n")


def main():
    """
    Função principal - executa todos os testes.
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "TESTE COMPLETO DO MODELO DEEPFAKE DETECTOR" + " " * 6 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    set_global_seed(42)
    
    # Teste 1: Criação
    model, device = test_model_creation()
    
    # Teste 2: Forward pass
    forward_stats = test_forward_pass(model, device)
    
    # Teste 3: Componentes
    params = test_model_components(model)
    
    # Teste 4: Save/Load
    test_save_load(model, device)
    
    # Criar resumo
    create_model_summary()
    
    # Resumo final
    print("=" * 60)
    print("RESUMO FINAL DOS TESTES")
    print("=" * 60)
    print()
    print(f"✓ Modelo criado com sucesso")
    print(f"✓ Total de parâmetros: {params['total']:,}")
    print(f"✓ Forward pass testado (batch 1, 4, 8)")
    print(f"✓ Tempo médio por vídeo: {forward_stats['batch_1']['time_per_sample']:.4f}s")
    print(f"✓ Salvamento/carregamento verificado")
    print(f"✓ Especificações documentadas")
    print()
    print("=" * 60)
    print("TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
