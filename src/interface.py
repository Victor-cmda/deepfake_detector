"""
Interface Gradio para demonstração do sistema.
Permite upload de vídeo e visualização de resultados com Grad-CAM.
"""

import os
import sys
from pathlib import Path
import time
import torch
import gradio as gr
import pandas as pd
from datetime import datetime
import numpy as np
from PIL import Image

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.utils import set_global_seed, get_device
from src.model import load_model
from src.preprocessing import preprocess_video
from src.gradcam import generate_video_gradcam
from facenet_pytorch import MTCNN


# Configurações globais
MODEL_PATH = 'models/model_best.pt'
LOG_PATH = 'outputs/reports/interface_log.csv'
HEATMAPS_DIR = 'outputs/heatmaps'

# Cache do modelo (carregado uma vez)
_model_cache = None
_device_cache = None
_mtcnn_cache = None


def initialize_model():
    """
    Inicializa e cacheia o modelo, device e MTCNN.
    
    Returns:
        model, device, mtcnn
    """
    global _model_cache, _device_cache, _mtcnn_cache
    
    if _model_cache is None:
        print("Inicializando modelo...")
        
        # Configurar seed
        set_global_seed(42)
        
        # Detectar device
        _device_cache = get_device()
        
        # Carregar modelo
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
        
        _model_cache, _ = load_model(MODEL_PATH, device=_device_cache)
        
        # Inicializar MTCNN (CPU para compatibilidade)
        _mtcnn_cache = MTCNN(device='cpu', post_process=False)
        
        print(f"✓ Modelo carregado (device: {_device_cache})")
    
    return _model_cache, _device_cache, _mtcnn_cache


def predict(video_path, num_frames=16, generate_gradcam=True):
    """
    Função principal de predição.
    
    Processa vídeo, gera predição e opcionalmente Grad-CAM.
    Loga execução em CSV.
    
    Args:
        video_path: Caminho do vídeo uploadado
        num_frames: Número de frames a processar
        generate_gradcam: Se True, gera visualização Grad-CAM
        
    Returns:
        label: String com label e probabilidade
        prob_text: String com probabilidade formatada
        gradcam_gallery: Lista de imagens Grad-CAM ou None
        log_text: Texto com informações do log
    """
    start_time = time.time()
    
    try:
        # Inicializar modelo
        model, device, mtcnn = initialize_model()
        
        # Pré-processar vídeo
        print(f"Processando vídeo: {video_path}")
        result = preprocess_video(video_path, num_frames=num_frames, mtcnn=mtcnn)
        
        if result is None or result[0] is None:
            return (
                "❌ ERRO: Não foi possível processar o vídeo",
                "Verifique se o vídeo contém faces detectáveis",
                None,
                "Processamento falhou"
            )
        
        video_tensor, detection_rate, preprocess_time = result
        
        # Preparar para inferência
        video_tensor_batch = video_tensor.unsqueeze(0).to(device)
        
        # Inferência
        model.eval()
        with torch.no_grad():
            output = model(video_tensor_batch)
            probabilidade_fake = output.item()
        
        # Classificação
        threshold = 0.5
        label = "FAKE" if probabilidade_fake >= threshold else "REAL"
        confidence = probabilidade_fake if label == "FAKE" else (1 - probabilidade_fake)
        
        # Tempo de inferência
        tempo_inferencia = time.time() - start_time
        
        # Gerar Grad-CAM se solicitado
        gradcam_images = None
        gradcam_info = ""
        
        if generate_gradcam:
            print("Gerando Grad-CAM...")
            gradcam_result = generate_video_gradcam(
                video_path=video_path,
                model_path=MODEL_PATH,
                num_frames=num_frames,
                output_dir=HEATMAPS_DIR,
                device=device,
                alpha=0.4
            )
            
            if gradcam_result:
                # Carregar imagens geradas
                gradcam_images = []
                for img_path in gradcam_result['heatmap_paths'][:8]:  # Mostrar primeiros 8 frames
                    if os.path.exists(img_path):
                        gradcam_images.append(img_path)
                
                gradcam_info = f"\nGrad-CAM: Atenção média = {gradcam_result['attention_mean']:.4f}"
        
        # Preparar resultado formatado
        label_emoji = "🎭 FAKE" if label == "FAKE" else "✅ REAL"
        label_text = f"{label_emoji}\n\nProbabilidade: {probabilidade_fake:.2%}"
        
        prob_text = (
            f"Probabilidade de ser FAKE: {probabilidade_fake:.2%}\n"
            f"Probabilidade de ser REAL: {(1-probabilidade_fake):.2%}\n"
            f"Confiança: {confidence:.2%}"
        )
        
        log_text = (
            f"📊 INFORMAÇÕES DA ANÁLISE\n"
            f"{'='*50}\n"
            f"Vídeo: {os.path.basename(video_path)}\n"
            f"Classificação: {label}\n"
            f"Probabilidade FAKE: {probabilidade_fake:.4f}\n"
            f"Threshold: {threshold}\n"
            f"Taxa detecção facial: {detection_rate:.1f}%\n"
            f"Tempo pré-processamento: {preprocess_time:.2f}s\n"
            f"Tempo inferência total: {tempo_inferencia:.2f}s\n"
            f"Device: {device}\n"
            f"Frames processados: {num_frames}"
            f"{gradcam_info}"
        )
        
        # Logar execução
        log_execution(
            video_path=video_path,
            label=label,
            probabilidade_fake=probabilidade_fake,
            tempo_inferencia=tempo_inferencia,
            detection_rate=detection_rate,
            num_frames=num_frames
        )
        
        return label_text, prob_text, gradcam_images, log_text
        
    except Exception as e:
        error_msg = f"❌ ERRO: {str(e)}"
        return error_msg, str(e), None, f"Erro durante processamento:\n{str(e)}"


def log_execution(video_path, label, probabilidade_fake, tempo_inferencia, detection_rate, num_frames):
    """
    Loga execução no arquivo CSV.
    
    Args:
        video_path: Caminho do vídeo
        label: Label predito (REAL/FAKE)
        probabilidade_fake: Probabilidade de ser fake
        tempo_inferencia: Tempo total de inferência
        detection_rate: Taxa de detecção facial
        num_frames: Número de frames processados
    """
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    
    # Preparar dados do log
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video_name': os.path.basename(video_path),
        'video_path': video_path,
        'prediction': label,
        'probabilidade_fake': probabilidade_fake,
        'probabilidade_real': 1 - probabilidade_fake,
        'tempo_inferencia': tempo_inferencia,
        'detection_rate': detection_rate,
        'num_frames': num_frames,
        'model': MODEL_PATH
    }
    
    # Verificar se arquivo existe
    if os.path.exists(LOG_PATH):
        # Append ao CSV existente
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        # Criar novo CSV
        df = pd.DataFrame([log_entry])
    
    # Salvar
    df.to_csv(LOG_PATH, index=False)
    print(f"✓ Execução logada em: {LOG_PATH}")


def create_interface():
    """
    Cria a interface Gradio.
    
    Returns:
        gr.Interface: Interface configurada
    """
    # CSS customizado
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-class {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
    }
    """
    
    # Descrição
    description = """
    ## 🎭 Detector de Deepfakes com Explicabilidade Visual
    
    Este sistema utiliza um modelo CNN-LSTM (ResNet-34 + BiLSTM) para detectar vídeos deepfake.
    
    **Funcionalidades:**
    - 🎥 Análise de vídeos com detecção facial automática
    - 📊 Probabilidade de autenticidade
    - 🔍 Visualização Grad-CAM (regiões importantes para decisão)
    - 📝 Log automático de execuções
    
    **Como usar:**
    1. Faça upload de um vídeo
    2. Ajuste o número de frames (opcional)
    3. Ative/desative visualização Grad-CAM
    4. Clique em "Analisar Vídeo"
    """
    
    # Exemplos
    examples = []
    for dataset in ['faceforensicspp', 'celebdf', 'wilddeepfake']:
        for label_dir in ['videos_real', 'videos_fake']:
            video_dir = f'data/{dataset}/{label_dir}'
            if os.path.exists(video_dir):
                videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
                if videos:
                    examples.append([os.path.join(video_dir, videos[0]), 16, True])
                    if len(examples) >= 3:
                        break
        if len(examples) >= 3:
            break
    
    # Criar interface
    with gr.Blocks(css=custom_css, title="Deepfake Detector") as interface:
        gr.Markdown("# 🎭 Sistema de Detecção de Deepfakes")
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Inputs
                video_input = gr.Video(
                    label="📹 Upload de Vídeo",
                    sources=["upload"]
                )
                
                num_frames_slider = gr.Slider(
                    minimum=8,
                    maximum=32,
                    value=16,
                    step=4,
                    label="Número de Frames",
                    info="Mais frames = análise mais precisa, mas mais lenta"
                )
                
                gradcam_checkbox = gr.Checkbox(
                    value=True,
                    label="Gerar visualização Grad-CAM",
                    info="Mostra regiões importantes para a decisão"
                )
                
                analyze_btn = gr.Button("🔍 Analisar Vídeo", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Outputs
                label_output = gr.Textbox(
                    label="Classificação",
                    elem_classes=["output-class"]
                )
                
                prob_output = gr.Textbox(
                    label="Probabilidades",
                    lines=3
                )
                
                log_output = gr.Textbox(
                    label="Informações da Análise",
                    lines=12
                )
        
        # Galeria de Grad-CAM
        gradcam_gallery = gr.Gallery(
            label="🔍 Visualização Grad-CAM (primeiros 8 frames)",
            columns=4,
            rows=2,
            height="auto",
            object_fit="contain"
        )
        
        # Conectar função
        analyze_btn.click(
            fn=predict,
            inputs=[video_input, num_frames_slider, gradcam_checkbox],
            outputs=[label_output, prob_output, gradcam_gallery, log_output]
        )
        
        # Exemplos
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[video_input, num_frames_slider, gradcam_checkbox],
                outputs=[label_output, prob_output, gradcam_gallery, log_output],
                fn=predict,
                cache_examples=False
            )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📚 Informações Técnicas
        
        **Modelo:** ResNet-34 (CNN) + BiLSTM (2 camadas, 256 unidades)  
        **Parâmetros:** 24.4M  
        **Datasets de Treino:** FaceForensics++  
        **Explicabilidade:** Grad-CAM (Gradient-weighted Class Activation Mapping)  
        
        **Logs:** Todas as execuções são registradas em `outputs/reports/interface_log.csv`
        """)
    
    return interface


def launch_interface(share=False, server_port=7860):
    """
    Lança a interface Gradio.
    
    Args:
        share: Se True, cria link público
        server_port: Porta do servidor
    """
    print("\n" + "="*70)
    print("INTERFACE GRADIO - DEEPFAKE DETECTOR")
    print("="*70 + "\n")
    
    # Verificar se modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: Modelo não encontrado em {MODEL_PATH}")
        print("Execute primeiro o treinamento (Tarefa 7).")
        return
    
    # Criar diretórios
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(HEATMAPS_DIR, exist_ok=True)
    
    # Inicializar modelo (cache)
    try:
        initialize_model()
    except Exception as e:
        print(f"❌ ERRO ao carregar modelo: {e}")
        return
    
    # Criar interface
    interface = create_interface()
    
    # Lançar
    print("\nIniciando servidor Gradio...")
    print(f"Porta: {server_port}")
    print(f"Share: {share}")
    print("\n" + "="*70)
    
    interface.launch(
        share=share,
        server_port=server_port,
        server_name="0.0.0.0",
        show_error=True
    )


def test_interface():
    """
    Função de teste para validar a interface sem lançá-la.
    """
    print("\n" + "="*70)
    print("TESTE DE INTERFACE")
    print("="*70 + "\n")
    
    # Verificar se modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"ERRO: Modelo não encontrado em {MODEL_PATH}")
        print("Execute primeiro o treinamento (Tarefa 7).")
        return
    
    # Buscar vídeo de teste
    test_video = None
    for dataset in ['faceforensicspp', 'celebdf', 'wilddeepfake']:
        for label_dir in ['videos_real', 'videos_fake']:
            video_dir = f'data/{dataset}/{label_dir}'
            if os.path.exists(video_dir):
                videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
                if videos:
                    test_video = os.path.join(video_dir, videos[0])
                    break
        if test_video:
            break
    
    if not test_video:
        print("ERRO: Nenhum vídeo de teste encontrado!")
        return
    
    print(f"Vídeo de teste: {test_video}\n")
    
    # Testar função predict
    label, prob, gradcam_imgs, log = predict(
        video_path=test_video,
        num_frames=16,
        generate_gradcam=True
    )
    
    print("Resultados:")
    print(f"  Label: {label}")
    print(f"  Probabilidades: {prob}")
    print(f"  Grad-CAM gerado: {len(gradcam_imgs) if gradcam_imgs else 0} imagens")
    print(f"\n{log}")
    
    # Verificar log CSV
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        print(f"\n✓ Log CSV criado: {LOG_PATH}")
        print(f"  Total de execuções: {len(df)}")
        print(f"\nÚltima entrada:")
        print(df.tail(1).to_string(index=False))
    
    print("\n" + "="*70)
    print("TESTE CONCLUÍDO COM SUCESSO")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Executar teste
    # test_interface()
    
    # Para lançar a interface, descomente a linha abaixo:
    launch_interface(share=False, server_port=7860)
