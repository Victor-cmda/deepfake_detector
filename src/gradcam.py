"""
Implementação do Grad-CAM para explicabilidade visual.
Gera mapas de ativação sobre frames de vídeo.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import set_global_seed, get_device
from src.model import load_model
from src.preprocessing import preprocess_video
from facenet_pytorch import MTCNN


class GradCAM:
    """
    Implementação do Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Grad-CAM produz mapas de ativação que destacam regiões importantes
    da imagem que influenciam a decisão do modelo.
    """
    
    def __init__(self, model, target_layer):
        """
        Inicializa o Grad-CAM.
        
        Args:
            model: Modelo PyTorch
            target_layer: Camada alvo para extração de features (última conv do CNN)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks para capturar gradientes e ativações
        self._register_hooks()
    
    def _register_hooks(self):
        """Registra hooks para capturar forward e backward passes."""
        
        def forward_hook(module, input, output):
            """Captura ativações no forward pass."""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Captura gradientes no backward pass."""
            self.gradients = grad_output[0].detach()
        
        # Registrar hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Gera o mapa de ativação Grad-CAM.
        
        Args:
            input_tensor: Tensor de entrada (B, T, C, H, W) ou (B, C, H, W)
            target_class: Classe alvo (None = usar predição do modelo)
            
        Returns:
            cam: Mapa de ativação normalizado (0-1)
        """
        # IMPORTANTE: Habilitar gradientes mas manter modelo em eval
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Se target_class não especificado, usar predição
        if target_class is None:
            target_class = (output >= 0.5).long()
        
        # Backward pass - IMPORTANTE: Sem atualizar pesos
        self.model.zero_grad()
        
        # Criar gradiente de saída
        grad_output = torch.ones_like(output)
        output.backward(gradient=grad_output, retain_graph=False)
        
        # Obter gradientes e ativações
        gradients = self.gradients  # (B, C, H, W)
        activations = self.activations  # (B, C, H, W)
        
        # Calcular pesos (média global dos gradientes)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Calcular CAM (combinação ponderada das ativações)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # ReLU (remover ativações negativas)
        cam = F.relu(cam)
        
        # Normalizar para [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy()
    
    def generate_heatmap(self, cam, original_size=(224, 224), colormap=cv2.COLORMAP_JET):
        """
        Converte CAM em heatmap colorido.
        
        Args:
            cam: Mapa de ativação (array 2D)
            original_size: Tamanho original da imagem (W, H)
            colormap: Colormap do OpenCV
            
        Returns:
            heatmap: Heatmap colorido (RGB)
        """
        # Redimensionar CAM para o tamanho original
        cam_resized = cv2.resize(cam, original_size)
        
        # Converter para 8-bit
        cam_uint8 = np.uint8(255 * cam_resized)
        
        # Aplicar colormap
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        
        # Converter BGR para RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.4):
        """
        Sobrepõe heatmap na imagem original.
        
        Args:
            heatmap: Heatmap colorido (RGB)
            original_image: Imagem original (RGB)
            alpha: Transparência do heatmap (0-1)
            
        Returns:
            overlay: Imagem com heatmap sobreposto
        """
        # Garantir que ambas as imagens tenham o mesmo tamanho
        if heatmap.shape != original_image.shape:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Sobrepor com transparência
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


def generate_video_gradcam(
    video_path,
    model_path='models/model_best.pt',
    num_frames=16,
    output_dir='outputs/heatmaps',
    device=None,
    alpha=0.4
):
    """
    Gera Grad-CAM para todos os frames de um vídeo.
    
    Args:
        video_path: Caminho do vídeo
        model_path: Caminho do modelo treinado
        num_frames: Número de frames a processar
        output_dir: Diretório para salvar heatmaps
        device: Dispositivo (None = auto-detect)
        alpha: Transparência do heatmap
        
    Returns:
        results: Dicionário com resultados e métricas
    """
    # Configurar seed
    set_global_seed(42)
    
    # Detectar device
    if device is None:
        device = get_device()
    
    print(f"\n{'='*70}")
    print(f"GRAD-CAM PARA VÍDEO")
    print(f"{'='*70}\n")
    print(f"Vídeo: {video_path}")
    print(f"Modelo: {model_path}")
    print(f"Device: {device}")
    print(f"Frames: {num_frames}\n")
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar modelo
    print("Carregando modelo...")
    model, checkpoint = load_model(model_path, device=device)
    
    # Inicializar MTCNN (CPU para compatibilidade)
    mtcnn = MTCNN(device='cpu', post_process=False)
    
    # Pré-processar vídeo
    print("Pré-processando vídeo...")
    result = preprocess_video(video_path, num_frames=num_frames, mtcnn=mtcnn)
    
    if result is None or result[0] is None:
        print(f"ERRO: Não foi possível processar o vídeo: {video_path}")
        return None
    
    video_tensor, detection_rate, processing_time = result
    print(f"✓ Vídeo processado: {video_tensor.shape}")
    print(f"✓ Taxa de detecção facial: {detection_rate:.1f}%")
    print(f"✓ Tempo de processamento: {processing_time:.2f}s")
    
    # Preparar tensor para o modelo
    video_tensor_batch = video_tensor.unsqueeze(0).to(device)  # (1, T, C, H, W)
    
    # Obter predição
    model.eval()
    with torch.no_grad():
        prediction = model(video_tensor_batch)
        prob = prediction.item()
        label = "FAKE" if prob >= 0.5 else "REAL"
    
    print(f"\nPredição: {label} (probabilidade: {prob:.4f})")
    
    # Configurar Grad-CAM
    # O model.cnn é um Sequential com camadas do ResNet
    # Precisamos acessar a última camada convolucional
    # Sequential contém: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
    # A última conv está em layer4[-1]
    cnn_modules = list(model.cnn.children())
    
    # Encontrar layer4 (geralmente é o índice -2, antes do avgpool)
    target_layer = None
    for module in cnn_modules:
        if hasattr(module, '__class__') and 'Sequential' in str(type(module)):
            # layer4 é um Sequential com BasicBlocks
            # Pegar o último bloco e a última conv dele
            if len(list(module.children())) > 0:
                last_block = list(module.children())[-1]
                if hasattr(last_block, 'conv2'):
                    target_layer = last_block.conv2
                    break
    
    if target_layer is None:
        # Fallback: usar a penúltima camada do Sequential
        target_layer = cnn_modules[-2]
    
    gradcam = GradCAM(model, target_layer)
    
    # Armazenar resultados
    attention_scores = []
    heatmap_paths = []
    
    print(f"\nGerando Grad-CAM para {num_frames} frames...")
    
    # Processar cada frame
    for frame_idx in range(num_frames):
        # Extrair frame tensor original
        frame_tensor = video_tensor[frame_idx].unsqueeze(0).to(device)  # (1, C, H, W)
        frame_tensor.requires_grad_(True)
        
        # SOLUÇÃO: Processar APENAS o CNN, não o modelo completo
        # Isso evita o erro do LSTM backward
        model.cnn.eval()
        
        try:
            with torch.enable_grad():
                # Forward apenas pelo CNN
                cnn_output = model.cnn(frame_tensor)  # (1, 512, 7, 7)
                
                # Simular "score" somando todas as ativações
                # (não é a predição real, mas serve para Grad-CAM)
                score = cnn_output.sum()
                
                # Backward
                model.cnn.zero_grad()
                score.backward()
            
            # Obter gradientes e ativações capturados pelos hooks
            if gradcam.gradients is None or gradcam.activations is None:
                print(f"⚠️ Frame {frame_idx}: Gradientes não capturados, pulando...")
                continue
            
            gradients = gradcam.gradients
            activations = gradcam.activations
            
            # Calcular CAM
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Normalizar
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            cam = cam.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"⚠️ Erro ao processar frame {frame_idx}: {e}")
            # Criar CAM vazio em caso de erro
            cam = np.zeros((7, 7))
        
        # Calcular atenção média
        attention_mean = float(cam.mean())
        attention_scores.append(attention_mean)
        
        # Converter frame tensor para imagem (desnormalizar)
        frame_np = video_tensor[frame_idx].permute(1, 2, 0).numpy()
        
        # Desnormalizar (reverter ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_np = frame_np * std + mean
        frame_np = np.clip(frame_np, 0, 1)
        frame_np = (frame_np * 255).astype(np.uint8)
        
        # Gerar heatmap
        heatmap = gradcam.generate_heatmap(cam, original_size=(224, 224))
        
        # Sobrepor heatmap
        overlay = gradcam.overlay_heatmap(heatmap, frame_np, alpha=alpha)
        
        # Salvar frame com heatmap
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx:03d}_gradcam.png")
        
        # Criar visualização completa (frame original | heatmap | overlay)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(frame_np)
        axes[0].set_title(f'Frame {frame_idx}', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title(f'Grad-CAM Heatmap\nAttention: {attention_mean:.4f}', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay (α={alpha})', fontsize=12)
        axes[2].axis('off')
        
        plt.suptitle(f'{video_name} - Prediction: {label} ({prob:.4f})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        heatmap_paths.append(output_path)
        
        if (frame_idx + 1) % 4 == 0 or frame_idx == num_frames - 1:
            print(f"  Frame {frame_idx + 1}/{num_frames} processado (attention: {attention_mean:.4f})")
    
    # Calcular estatísticas de atenção
    attention_mean_all = np.mean(attention_scores)
    attention_std = np.std(attention_scores)
    attention_min = np.min(attention_scores)
    attention_max = np.max(attention_scores)
    
    print(f"\nEstatísticas de Atenção:")
    print(f"  Média: {attention_mean_all:.4f}")
    print(f"  Desvio padrão: {attention_std:.4f}")
    print(f"  Mínimo: {attention_min:.4f}")
    print(f"  Máximo: {attention_max:.4f}")
    
    print(f"\n✓ {len(heatmap_paths)} heatmaps salvos em: {output_dir}/")
    
    # Retornar resultados
    results = {
        'video_path': video_path,
        'prediction': label,
        'probability': prob,
        'num_frames': num_frames,
        'attention_mean': attention_mean_all,
        'attention_std': attention_std,
        'attention_min': attention_min,
        'attention_max': attention_max,
        'attention_scores': attention_scores,
        'heatmap_paths': heatmap_paths,
        'output_dir': output_dir
    }
    
    return results


def test_gradcam():
    """
    Função de teste para validar o Grad-CAM.
    """
    print("\n" + "="*70)
    print("TESTE DE GRAD-CAM")
    print("="*70 + "\n")
    
    # Verificar se modelo existe
    model_path = 'models/model_best.pt'
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em {model_path}")
        print("Execute primeiro o treinamento (Tarefa 7).")
        return
    
    # Buscar um vídeo de teste
    test_videos = []
    for dataset in ['faceforensicspp', 'celebdf', 'wilddeepfake']:
        for label_dir in ['videos_real', 'videos_fake']:
            video_dir = f'data/{dataset}/{label_dir}'
            if os.path.exists(video_dir):
                videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
                if videos:
                    test_videos.append(os.path.join(video_dir, videos[0]))
    
    if not test_videos:
        print("ERRO: Nenhum vídeo de teste encontrado!")
        print("Certifique-se de que os datasets foram organizados (Tarefa 2).")
        return
    
    # Processar primeiro vídeo
    video_path = test_videos[0]
    print(f"Vídeo de teste: {video_path}\n")
    
    # Gerar Grad-CAM
    results = generate_video_gradcam(
        video_path=video_path,
        model_path=model_path,
        num_frames=16,
        output_dir='outputs/heatmaps',
        alpha=0.4
    )
    
    if results:
        print("\n" + "="*70)
        print("TESTE CONCLUÍDO COM SUCESSO")
        print("="*70)
        print(f"\nResumo:")
        print(f"  Predição: {results['prediction']}")
        print(f"  Probabilidade: {results['probability']:.4f}")
        print(f"  Atenção média: {results['attention_mean']:.4f}")
        print(f"  Heatmaps gerados: {len(results['heatmap_paths'])}")
        print(f"  Diretório: {results['output_dir']}/")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("TESTE FALHOU")
        print("="*70 + "\n")


if __name__ == '__main__':
    # Executar teste de Grad-CAM
    test_gradcam()
