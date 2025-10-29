"""
Script para testar correções da interface Gradio.
"""

import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

print("=" * 70)
print("TESTE DE CORREÇÕES DA INTERFACE")
print("=" * 70)
print()

# Verificar se modelo existe
model_path = 'models/model_best.pt'
if not os.path.exists(model_path):
    print(f"❌ ERRO: Modelo não encontrado em {model_path}")
    print("Execute primeiro o treinamento.")
    sys.exit(1)

print(f"✓ Modelo encontrado: {model_path}")

# Testar importação
try:
    from src.interface import initialize_model, predict
    print("✓ Módulos importados com sucesso")
except Exception as e:
    print(f"❌ ERRO ao importar módulos: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Testar inicialização do modelo
print("\nInicializando modelo...")
try:
    model, device, mtcnn = initialize_model()
    print(f"✓ Modelo inicializado com sucesso")
    print(f"  - Device: {device}")
    print(f"  - MTCNN: {'OK' if mtcnn else 'Erro'}")
except Exception as e:
    print(f"❌ ERRO ao inicializar modelo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Testar função predict com None (simulando sem upload)
print("\nTestando predict com None (sem vídeo)...")
try:
    label, prob, gradcam, log = predict(None, 16, False)
    print(f"✓ Teste sem vídeo passou")
    print(f"  - Label: {label[:50]}...")
except Exception as e:
    print(f"❌ ERRO: {e}")
    import traceback
    traceback.print_exc()

# Buscar vídeo de teste
print("\nBuscando vídeo de teste...")
test_video = None

# Buscar nos datasets
for dataset_dir in ['FaceForensics++', 'Celeb-DF-v2', 'WildDeepfake']:
    for label_dir in ['videos_real', 'videos_fake']:
        video_dir = f'data/{dataset_dir}/{label_dir}'
        if os.path.exists(video_dir):
            videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            if videos:
                test_video = os.path.join(video_dir, videos[0])
                print(f"✓ Vídeo de teste encontrado: {test_video}")
                break
    if test_video:
        break

if not test_video:
    print("⚠️ Nenhum vídeo de teste encontrado")
    print("\nCriando vídeo de teste simples...")
    
    # Criar vídeo de teste usando OpenCV
    import cv2
    import numpy as np
    
    os.makedirs('data/test', exist_ok=True)
    test_video = 'data/test/test_video.mp4'
    
    # Criar vídeo simples com face simulada
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(test_video, fourcc, 30.0, (640, 480))
    
    for i in range(60):  # 2 segundos a 30 fps
        # Criar frame com gradiente
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Adicionar círculo simulando face
        center = (320, 240)
        radius = 100
        color = (255, 200, 200)  # Tom de pele
        cv2.circle(frame, center, radius, color, -1)
        
        # Adicionar olhos
        cv2.circle(frame, (280, 220), 15, (0, 0, 0), -1)
        cv2.circle(frame, (360, 220), 15, (0, 0, 0), -1)
        
        # Adicionar boca
        cv2.ellipse(frame, (320, 280), (30, 20), 0, 0, 180, (255, 0, 0), -1)
        
        # Adicionar texto
        cv2.putText(frame, f"Test Frame {i+1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Vídeo de teste criado: {test_video}")

# Testar função predict com vídeo real
if test_video and os.path.exists(test_video):
    print(f"\nTestando predict com vídeo: {os.path.basename(test_video)}...")
    print("(Grad-CAM desabilitado para teste rápido)")
    
    try:
        label, prob, gradcam, log = predict(test_video, 8, False)
        
        print(f"✓ Predição bem-sucedida!")
        print(f"\nResultados:")
        print(f"  - Label: {label}")
        print(f"  - Probabilidades:\n{prob}")
        print(f"\n{log}")
        
    except Exception as e:
        print(f"❌ ERRO na predição: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("TESTE CONCLUÍDO")
print("=" * 70)
print()
print("Correções aplicadas:")
print("  ✓ Modo de avaliação do modelo corrigido (cudnn RNN)")
print("  ✓ Validação de input (vídeo None)")
print("  ✓ Exemplos removidos (problemas de codec)")
print("  ✓ Formato de vídeo especificado (mp4)")
print()
print("Próximo passo: Reiniciar a interface Gradio")
print("  python src/interface.py")
print()
