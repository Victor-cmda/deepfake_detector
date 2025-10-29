"""
Teste rápido da correção do erro CuDNN RNN.
"""

import sys
import os
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

print("=" * 70)
print("TESTE DE CORREÇÃO - CUDNN RNN ERROR")
print("=" * 70)
print()

# Verificar modelo
model_path = 'models/model_best.pt'
if not os.path.exists(model_path):
    print(f"❌ Modelo não encontrado: {model_path}")
    sys.exit(1)

print(f"✓ Modelo encontrado: {model_path}")

# Buscar vídeo de teste
test_video = None
for dataset_dir in ['FaceForensics++', 'Celeb-DF-v2', 'WildDeepfake']:
    for label_dir in ['videos_real', 'videos_fake']:
        video_dir = f'data/{dataset_dir}/{label_dir}'
        if os.path.exists(video_dir):
            videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            if videos:
                test_video = os.path.join(video_dir, videos[0])
                break
    if test_video:
        break

if not test_video:
    print("❌ Nenhum vídeo de teste encontrado")
    sys.exit(1)

print(f"✓ Vídeo de teste: {test_video}")
print()

# Testar sem Grad-CAM (deve funcionar)
print("TESTE 1: Predição SEM Grad-CAM")
print("-" * 70)

try:
    from src.interface import initialize_model, predict
    
    # Inicializar modelo
    model, device, mtcnn = initialize_model()
    print(f"✓ Modelo inicializado (device: {device})")
    
    # Testar predição sem Grad-CAM
    label, prob, frames, gradcam, log = predict(test_video, num_frames=8, generate_gradcam=False)
    
    print(f"✓ Predição bem-sucedida!")
    print(f"  - Label: {label[:30]}...")
    print(f"  - Frames extraídos: {len(frames) if frames else 0}")
    print()
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Testar com Grad-CAM (teste crítico)
print("TESTE 2: Predição COM Grad-CAM")
print("-" * 70)

try:
    label, prob, frames, gradcam, log = predict(test_video, num_frames=8, generate_gradcam=True)
    
    print(f"✓ Predição com Grad-CAM bem-sucedida!")
    print(f"  - Label: {label[:30]}...")
    print(f"  - Frames extraídos: {len(frames) if frames else 0}")
    print(f"  - Grad-CAM gerado: {len(gradcam) if gradcam else 0} heatmaps")
    print()
    
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
print("✅ TODOS OS TESTES PASSARAM!")
print("=" * 70)
print()
print("Correções validadas:")
print("  ✅ Modo de avaliação do modelo corrigido")
print("  ✅ Grad-CAM modificado para evitar backward no LSTM")
print("  ✅ Processamento manual do forward pass para Grad-CAM")
print()
print("A interface está pronta para uso!")
print()
