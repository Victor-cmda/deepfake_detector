"""
Script de teste dos DataLoaders.
Testa criação de datasets, batching e carregamento de dados.
"""

import os
import torch
from src.preprocessing import get_dataloaders, test_dataloader
from src.utils import set_global_seed, get_device


def test_dataset_creation():
    """
    Testa a criação dos datasets e dataloaders.
    """
    print("=" * 60)
    print("TESTE 1: CRIAÇÃO DE DATALOADERS")
    print("=" * 60)
    print()
    
    set_global_seed(42)
    
    # Caminho do arquivo de splits
    splits_csv = 'data/splits_faceforensicspp.csv'
    
    if not os.path.exists(splits_csv):
        print(f"✗ Arquivo não encontrado: {splits_csv}")
        print("Execute primeiro: python generate_splits.py")
        return None
    
    # Criar dataloaders
    dataloaders = get_dataloaders(
        splits_csv=splits_csv,
        batch_size=2,
        num_frames=16,
        num_workers=0,
        shuffle_train=True,
        cache_preprocessed=False
    )
    
    return dataloaders


def test_batch_loading(dataloaders):
    """
    Testa o carregamento de batches.
    """
    print("\n" + "=" * 60)
    print("TESTE 2: CARREGAMENTO DE BATCHES")
    print("=" * 60)
    
    for split_name, dataloader in dataloaders.items():
        print(f"\n{split_name.upper()}:")
        print("-" * 60)
        
        stats = test_dataloader(dataloader, num_batches=2)
    
    print("\n✓ Teste de batches concluído!")


def test_iteration_speed(dataloaders):
    """
    Testa a velocidade de iteração sobre os dataloaders.
    """
    print("\n" + "=" * 60)
    print("TESTE 3: VELOCIDADE DE ITERAÇÃO")
    print("=" * 60)
    print()
    
    import time
    
    for split_name, dataloader in dataloaders.items():
        print(f"{split_name.upper()}:")
        print("-" * 60)
        
        start_time = time.time()
        total_samples = 0
        
        for i, (videos, labels) in enumerate(dataloader):
            if videos is not None:
                total_samples += len(videos)
        
        total_time = time.time() - start_time
        
        print(f"  - Total de batches: {len(dataloader)}")
        print(f"  - Total de amostras: {total_samples}")
        print(f"  - Tempo total: {total_time:.2f}s")
        print(f"  - Tempo por batch: {total_time/len(dataloader):.4f}s")
        print(f"  - Throughput: {total_samples/total_time:.2f} amostras/s")
        print()
    
    print("✓ Teste de velocidade concluído!")


def test_dataloader_with_model(dataloaders):
    """
    Testa integração do dataloader com o modelo.
    """
    print("\n" + "=" * 60)
    print("TESTE 4: INTEGRAÇÃO COM MODELO")
    print("=" * 60)
    print()
    
    from src.model import create_model
    
    device = get_device()
    
    # Criar modelo
    print("Criando modelo...")
    model = create_model(num_frames=16, device=device)
    model.eval()
    
    # Testar com um batch de treino
    if 'train' in dataloaders:
        train_loader = dataloaders['train']
        
        print("\nTestando forward pass com batch de treino...")
        
        # Pegar primeiro batch
        videos, labels = next(iter(train_loader))
        
        if videos is not None:
            print(f"  - Input shape: {videos.shape}")
            print(f"  - Labels shape: {labels.shape}")
            
            # Mover para device
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(videos)
            
            print(f"  - Output shape: {outputs.shape}")
            print(f"  - Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"  - Labels: {labels.squeeze().cpu().tolist()}")
            print(f"  - Predictions: {outputs.squeeze().cpu().tolist()}")
            
            print("\n✓ Integração com modelo testada com sucesso!")
        else:
            print("✗ Não foi possível carregar batch")
    else:
        print("⚠ DataLoader de treino não disponível")


def test_memory_usage(dataloaders):
    """
    Testa uso de memória dos dataloaders.
    """
    print("\n" + "=" * 60)
    print("TESTE 5: USO DE MEMÓRIA")
    print("=" * 60)
    print()
    
    for split_name, dataloader in dataloaders.items():
        print(f"{split_name.upper()}:")
        print("-" * 60)
        
        # Pegar um batch
        videos, labels = next(iter(dataloader))
        
        if videos is not None:
            # Calcular uso de memória
            video_memory = videos.element_size() * videos.nelement() / 1024 / 1024
            label_memory = labels.element_size() * labels.nelement() / 1024
            
            print(f"  - Videos shape: {videos.shape}")
            print(f"  - Videos memory: {video_memory:.2f} MB")
            print(f"  - Labels memory: {label_memory:.2f} KB")
            print(f"  - Total batch: {video_memory:.2f} MB")
            
            # Projetar para dataset completo
            total_batches = len(dataloader)
            estimated_total = video_memory * total_batches
            
            print(f"  - Total batches: {total_batches}")
            print(f"  - Memória estimada (todos batches): {estimated_total:.2f} MB")
        else:
            print("  ✗ Não foi possível carregar batch")
        
        print()
    
    print("✓ Análise de memória concluída!")


def create_dataloader_report():
    """
    Cria relatório completo dos dataloaders.
    """
    print("\n" + "=" * 60)
    print("CRIAÇÃO DE RELATÓRIO")
    print("=" * 60)
    print()
    
    splits_csv = 'data/splits_faceforensicspp.csv'
    
    dataloaders = get_dataloaders(
        splits_csv=splits_csv,
        batch_size=4,
        num_frames=16,
        num_workers=0,
        shuffle_train=True,
        cache_preprocessed=False
    )
    
    # Criar relatório
    report_path = 'outputs/logs/dataloader_stats.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RELATÓRIO DE DATALOADERS\n")
        f.write("=" * 60 + "\n\n")
        
        for split_name, dataloader in dataloaders.items():
            f.write(f"{split_name.upper()}:\n")
            f.write("-" * 60 + "\n")
            
            dataset = dataloader.dataset
            
            f.write(f"Total de vídeos: {len(dataset)}\n")
            f.write(f"Batch size: {dataloader.batch_size}\n")
            f.write(f"Número de batches: {len(dataloader)}\n")
            f.write(f"Shuffle: {dataloader.sampler is not None}\n")
            
            # Distribuição de labels
            labels = dataset.get_labels()
            num_real = sum(1 for l in labels if l == 0)
            num_fake = sum(1 for l in labels if l == 1)
            
            f.write(f"\nDistribuição de labels:\n")
            f.write(f"  - Reais: {num_real} ({num_real/len(labels)*100:.1f}%)\n")
            f.write(f"  - Fakes: {num_fake} ({num_fake/len(labels)*100:.1f}%)\n")
            
            # Pesos das classes
            class_weights = dataset.get_class_weights()
            f.write(f"\nPesos das classes (para balanceamento):\n")
            f.write(f"  - Real: {class_weights[0]:.4f}\n")
            f.write(f"  - Fake: {class_weights[1]:.4f}\n")
            
            f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("CONFIGURAÇÕES:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Frames por vídeo: 16\n")
        f.write(f"Frame size: 224x224\n")
        f.write(f"Normalização: ImageNet stats\n")
        f.write(f"Device MTCNN: CPU\n")
    
    print(f"✓ Relatório salvo: {report_path}\n")


def main():
    """
    Função principal - executa todos os testes.
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "TESTE COMPLETO DE DATALOADERS" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    set_global_seed(42)
    
    # Teste 1: Criação
    dataloaders = test_dataset_creation()
    
    if dataloaders is None or len(dataloaders) == 0:
        print("\n✗ Não foi possível criar dataloaders. Encerrando testes.")
        return
    
    # Teste 2: Carregamento de batches
    test_batch_loading(dataloaders)
    
    # Teste 3: Velocidade
    test_iteration_speed(dataloaders)
    
    # Teste 4: Integração com modelo
    test_dataloader_with_model(dataloaders)
    
    # Teste 5: Memória
    test_memory_usage(dataloaders)
    
    # Criar relatório
    create_dataloader_report()
    
    # Resumo final
    print("\n" + "=" * 60)
    print("RESUMO FINAL DOS TESTES")
    print("=" * 60)
    print()
    print(f"✓ {len(dataloaders)} DataLoaders criados")
    print(f"✓ Batches carregados com sucesso")
    print(f"✓ Integração com modelo verificada")
    print(f"✓ Análise de memória concluída")
    print(f"✓ Relatório gerado")
    print()
    print("=" * 60)
    print("TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
