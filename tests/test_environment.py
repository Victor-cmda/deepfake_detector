#!/usr/bin/env python
"""
Script de teste do ambiente - Valida todas as dependências e GPU.
"""

import sys
from pathlib import Path

print("\n" + "="*80)
print("TESTE DE AMBIENTE - DEEPFAKE DETECTOR")
print("="*80 + "\n")

# 1. Teste de imports básicos
print("1. Testando imports básicos...")
try:
    import torch
    import torchvision
    import cv2
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import matplotlib
    matplotlib.use('Agg')  # Backend não-interativo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gradio as gr
    from mtcnn import MTCNN
    print("   ✅ Todos os imports básicos OK")
except Exception as e:
    print(f"   ❌ Erro nos imports: {e}")
    sys.exit(1)

# 2. Teste PyTorch + CUDA
print("\n2. Testando PyTorch e CUDA...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU device: {torch.cuda.get_device_name(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Teste de operação CUDA
    try:
        x = torch.rand(100, 100).cuda()
        y = torch.rand(100, 100).cuda()
        z = torch.matmul(x, y)
        print(f"   ✅ Operação CUDA executada com sucesso!")
    except Exception as e:
        print(f"   ❌ Erro na operação CUDA: {e}")
else:
    print("   ⚠️  CUDA não disponível - o treinamento será MUITO lento!")

# 3. Teste OpenCV
print("\n3. Testando OpenCV...")
print(f"   OpenCV version: {cv2.__version__}")
try:
    # Criar imagem de teste
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 255, 0)
    print("   ✅ OpenCV funcionando")
except Exception as e:
    print(f"   ❌ Erro no OpenCV: {e}")

# 4. Teste MTCNN
print("\n4. Testando MTCNN (detecção facial)...")
try:
    detector = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu')
    print("   ✅ MTCNN inicializado")
except Exception as e:
    print(f"   ❌ Erro no MTCNN: {e}")

# 5. Verificar arquivos de dados
print("\n5. Verificando arquivos de dados...")
data_path = Path("data")
splits_file = data_path / "splits_faceforensicspp.csv"

if splits_file.exists():
    df = pd.read_csv(splits_file)
    print(f"   ✅ Arquivo de splits encontrado: {len(df)} vídeos")
    print(f"      - Train: {len(df[df['split']=='train'])} vídeos")
    print(f"      - Val: {len(df[df['split']=='val'])} vídeos")
    print(f"      - Test: {len(df[df['split']=='test'])} vídeos")
    
    # Verificar vídeos reais
    real_dir = Path("data/FaceForensics++/videos_real")
    fake_dir = Path("data/FaceForensics++/videos_fake")
    
    real_count = len(list(real_dir.glob("*.mp4"))) if real_dir.exists() else 0
    fake_count = len(list(fake_dir.glob("*.mp4"))) if fake_dir.exists() else 0
    
    print(f"      - Vídeos reais encontrados: {real_count}")
    print(f"      - Vídeos fake encontrados: {fake_count}")
else:
    print(f"   ⚠️  Arquivo de splits não encontrado: {splits_file}")

# 6. Verificar estrutura de outputs
print("\n6. Verificando estrutura de outputs...")
output_dirs = [
    "outputs/logs",
    "outputs/reports", 
    "outputs/figures",
    "outputs/heatmaps"
]

all_exist = True
for dir_path in output_dirs:
    path = Path(dir_path)
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {dir_path}")
    if not exists:
        all_exist = False
        path.mkdir(parents=True, exist_ok=True)
        print(f"      → Criado")

# 7. Teste do modelo
print("\n7. Testando importação do modelo...")
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from src.model import DeepfakeDetector
    
    model = DeepfakeDetector()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Teste forward pass
    dummy_input = torch.rand(1, 3, 16, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   ✅ Modelo carregado e testado")
    print(f"      - Input shape: {dummy_input.shape}")
    print(f"      - Output shape: {output.shape}")
    print(f"      - Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"   ❌ Erro ao testar modelo: {e}")
    import traceback
    traceback.print_exc()

# 8. Resumo final
print("\n" + "="*80)
print("RESUMO DO TESTE")
print("="*80)
print(f"✅ Python: {sys.version.split()[0]}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"{'✅' if torch.cuda.is_available() else '⚠️ '} CUDA: {torch.cuda.is_available()}")
print(f"✅ OpenCV: {cv2.__version__}")
print(f"✅ Dados: {splits_file.exists()}")
print(f"✅ Estrutura: OK")
print("="*80)

if torch.cuda.is_available():
    print("\n🚀 Ambiente configurado corretamente! Pronto para treinamento com GPU.")
else:
    print("\n⚠️  Ambiente OK, mas sem GPU. Treinamento será lento.")

print("="*80 + "\n")
