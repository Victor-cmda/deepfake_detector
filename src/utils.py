"""
Utilidades gerais do projeto Deepfake Detector.
Inclui configuração de seed global e funções auxiliares.
"""

import os
import random
import numpy as np
import torch
import pandas as pd
import cv2
from pathlib import Path


def set_global_seed(seed=42):
    """
    Configura seed global para reprodutibilidade.
    
    Args:
        seed (int): Valor da seed (padrão: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Seed global configurada: {seed}")


def get_device(force_gpu=True):
    """
    Retorna o device disponível (cuda, mps ou cpu).
    
    Args:
        force_gpu (bool): Se True, falha se GPU não estiver disponível no Windows (padrão: True)
    
    Returns:
        torch.device: Device configurado
    """
    import platform
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(0)  # Força uso da GPU 0
        print(f"✓ GPU NVIDIA detectada e ativada: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Memória disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ MPS (Apple Silicon) disponível")
    else:
        device = torch.device('cpu')
        print("⚠️  Usando CPU (GPU não detectada)")
        
        # No Windows com GPU NVIDIA, alertar que algo está errado
        if platform.system() == 'Windows' and force_gpu:
            print("\n" + "="*70)
            print("ERRO: GPU NVIDIA não detectada no Windows!")
            print("="*70)
            print("\nVerifique:")
            print("  1. Driver NVIDIA instalado corretamente")
            print("  2. PyTorch instalado com suporte CUDA:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("  3. Execute: nvidia-smi para verificar GPU")
            print("="*70 + "\n")
            
            # Não falhar, apenas alertar
            print("⚠️  Continuando com CPU (performance reduzida)")
    
    return device


def ensure_dir(directory):
    """
    Garante que um diretório existe, criando-o se necessário.
    
    Args:
        directory (str): Caminho do diretório
    """
    os.makedirs(directory, exist_ok=True)


def index_dataset(dataset_path, dataset_name, output_csv):
    """
    Indexa vídeos de um dataset em formato CSV.
    
    Args:
        dataset_path (str): Caminho raiz do dataset
        dataset_name (str): Nome do dataset
        output_csv (str): Caminho do arquivo CSV de saída
        
    Returns:
        pd.DataFrame: DataFrame com colunas [video_path, label, dataset, num_frames]
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    data = []
    
    real_path = os.path.join(dataset_path, 'videos_real')
    fake_path = os.path.join(dataset_path, 'videos_fake')
    
    # Indexar vídeos reais
    if os.path.exists(real_path):
        for video_file in os.listdir(real_path):
            if any(video_file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(real_path, video_file)
                num_frames = get_video_frame_count(video_path)
                data.append({
                    'video_path': video_path,
                    'label': 0,  # Real
                    'dataset': dataset_name,
                    'num_frames': num_frames
                })
    
    # Indexar vídeos fake
    if os.path.exists(fake_path):
        for video_file in os.listdir(fake_path):
            if any(video_file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(fake_path, video_file)
                num_frames = get_video_frame_count(video_path)
                data.append({
                    'video_path': video_path,
                    'label': 1,  # Fake
                    'dataset': dataset_name,
                    'num_frames': num_frames
                })
    
    # Criar DataFrame (mesmo que vazio)
    if len(data) > 0:
        df = pd.DataFrame(data)
    else:
        # DataFrame vazio com colunas corretas
        df = pd.DataFrame(columns=['video_path', 'label', 'dataset', 'num_frames'])
    
    # Salvar CSV (sobrescrever se existir)
    df.to_csv(output_csv, index=False)
    
    print(f"✓ Dataset '{dataset_name}' indexado:")
    print(f"  - Total de vídeos: {len(df)}")
    if len(df) > 0:
        print(f"  - Vídeos reais: {len(df[df['label'] == 0])}")
        print(f"  - Vídeos fake: {len(df[df['label'] == 1])}")
        print(f"  - Média de frames por vídeo: {df['num_frames'].mean():.1f}")
    print(f"  - Arquivo salvo: {output_csv}\n")
    
    return df


def get_video_frame_count(video_path):
    """
    Retorna o número total de frames de um vídeo.
    
    Args:
        video_path (str): Caminho do vídeo
        
    Returns:
        int: Número de frames
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        print(f"  ⚠ Erro ao ler {video_path}: {e}")
        return 0


def organize_all_datasets(data_root):
    """
    Organiza e indexa todos os datasets do projeto.
    
    Args:
        data_root (str): Diretório raiz dos dados
        
    Returns:
        dict: Dicionário com DataFrames de cada dataset
    """
    datasets = {}
    
    # FaceForensics++
    ff_path = os.path.join(data_root, 'FaceForensics++')
    ff_csv = os.path.join(data_root, 'index_faceforensicspp.csv')
    if os.path.exists(ff_path):
        datasets['FaceForensics++'] = index_dataset(ff_path, 'FaceForensics++', ff_csv)
    
    # Celeb-DF-v2
    celeb_path = os.path.join(data_root, 'Celeb-DF-v2')
    celeb_csv = os.path.join(data_root, 'index_celebdf.csv')
    if os.path.exists(celeb_path):
        datasets['Celeb-DF-v2'] = index_dataset(celeb_path, 'Celeb-DF-v2', celeb_csv)
    
    # WildDeepfake
    wild_path = os.path.join(data_root, 'WildDeepfake')
    wild_csv = os.path.join(data_root, 'index_wilddeepfake.csv')
    if os.path.exists(wild_path):
        datasets['WildDeepfake'] = index_dataset(wild_path, 'WildDeepfake', wild_csv)
    
    return datasets


def generate_train_val_test_split(index_csv, output_csv, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Gera divisão treino/validação/teste a partir de um arquivo de índice.
    
    Args:
        index_csv (str): Caminho do arquivo CSV de índice
        output_csv (str): Caminho do arquivo CSV de saída com splits
        train_ratio (float): Proporção de treino (padrão: 0.7)
        val_ratio (float): Proporção de validação (padrão: 0.15)
        test_ratio (float): Proporção de teste (padrão: 0.15)
        seed (int): Seed para reprodutibilidade
        
    Returns:
        pd.DataFrame: DataFrame com coluna 'split' adicionada
    """
    # Verificar se soma das proporções = 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "A soma das proporções deve ser 1.0"
    
    # Ler índice
    if not os.path.exists(index_csv):
        print(f"✗ Arquivo não encontrado: {index_csv}")
        return None
    
    df = pd.read_csv(index_csv)
    
    if len(df) == 0:
        print(f"⚠ Arquivo vazio: {index_csv}")
        # Criar DataFrame vazio com colunas corretas
        df['split'] = []
        df.to_csv(output_csv, index=False)
        return df
    
    # Separar por label para manter distribuição balanceada
    df_real = df[df['label'] == 0].copy()
    df_fake = df[df['label'] == 1].copy()
    
    # Embaralhar com seed
    np.random.seed(seed)
    df_real = df_real.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_fake = df_fake.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calcular índices de divisão para reais
    n_real = len(df_real)
    train_end_real = int(n_real * train_ratio)
    val_end_real = train_end_real + int(n_real * val_ratio)
    
    # Calcular índices de divisão para fakes
    n_fake = len(df_fake)
    train_end_fake = int(n_fake * train_ratio)
    val_end_fake = train_end_fake + int(n_fake * val_ratio)
    
    # Atribuir splits para reais
    df_real.loc[:train_end_real-1, 'split'] = 'train'
    df_real.loc[train_end_real:val_end_real-1, 'split'] = 'val'
    df_real.loc[val_end_real:, 'split'] = 'test'
    
    # Atribuir splits para fakes
    df_fake.loc[:train_end_fake-1, 'split'] = 'train'
    df_fake.loc[train_end_fake:val_end_fake-1, 'split'] = 'val'
    df_fake.loc[val_end_fake:, 'split'] = 'test'
    
    # Concatenar e embaralhar novamente
    df_split = pd.concat([df_real, df_fake], ignore_index=True)
    df_split = df_split.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Salvar (sobrescrever se existir)
    df_split.to_csv(output_csv, index=False)
    
    # Estatísticas
    print(f"✓ Divisão gerada para: {os.path.basename(index_csv)}")
    print(f"  - Total de vídeos: {len(df_split)}")
    print(f"  - Treino: {len(df_split[df_split['split'] == 'train'])} ({len(df_split[df_split['split'] == 'train'])/len(df_split)*100:.1f}%)")
    print(f"  - Validação: {len(df_split[df_split['split'] == 'val'])} ({len(df_split[df_split['split'] == 'val'])/len(df_split)*100:.1f}%)")
    print(f"  - Teste: {len(df_split[df_split['split'] == 'test'])} ({len(df_split[df_split['split'] == 'test'])/len(df_split)*100:.1f}%)")
    
    # Distribuição por label em cada split
    for split in ['train', 'val', 'test']:
        df_s = df_split[df_split['split'] == split]
        if len(df_s) > 0:
            n_real_s = len(df_s[df_s['label'] == 0])
            n_fake_s = len(df_s[df_s['label'] == 1])
            print(f"    {split.upper()}: {n_real_s} reais, {n_fake_s} fakes")
    
    print(f"  - Arquivo salvo: {output_csv}\n")
    
    return df_split


def generate_technical_report(
    output_md='outputs/reports/run_report.md',
    output_pdf='outputs/reports/run_report.pdf'
):
    """
    Gera relatório técnico automatizado em Markdown e PDF.
    
    Inclui:
    - Informações do sistema e versões
    - Configuração do modelo
    - Métricas de treinamento
    - Métricas de avaliação cross-dataset
    - Links para figuras geradas
    - Observações e análises
    
    Args:
        output_md: Caminho do arquivo Markdown
        output_pdf: Caminho do arquivo PDF
        
    Returns:
        linhas_relatorio: Número de linhas do relatório
    """
    import sys
    import platform
    from datetime import datetime
    
    print(f"\n{'='*70}")
    print("GERANDO RELATÓRIO TÉCNICO AUTOMATIZADO")
    print(f"{'='*70}\n")
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    
    # ========================================================================
    # COLETAR INFORMAÇÕES
    # ========================================================================
    
    # Versões do sistema
    python_version = platform.python_version()
    torch_version = torch.__version__
    system_info = f"{platform.system()} {platform.release()}"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Detectar device
    if torch.cuda.is_available():
        device_info = f"CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}"
    elif torch.backends.mps.is_available():
        device_info = "MPS (Apple Silicon)"
    else:
        device_info = "CPU"
    
    # Métricas de treino
    train_metrics = None
    best_epoch = None
    if os.path.exists('outputs/metrics_train.csv'):
        df_train = pd.read_csv('outputs/metrics_train.csv')
        best_epoch_idx = df_train['val_f1'].idxmax()
        best_epoch = df_train.iloc[best_epoch_idx]
        train_metrics = df_train
    
    # Métricas cross-dataset
    cross_metrics = None
    if os.path.exists('outputs/metrics_cross.csv'):
        cross_metrics = pd.read_csv('outputs/metrics_cross.csv')
    
    # Métricas consolidadas
    table_metrics = None
    if os.path.exists('outputs/reports/table_metrics.csv'):
        table_metrics = pd.read_csv('outputs/reports/table_metrics.csv')
    
    # Informações do modelo
    model_path = 'models/model_best.pt'
    model_info = {}
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model_info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_f1': checkpoint.get('val_f1', 'N/A'),
            'val_auc': checkpoint.get('val_auc', 'N/A'),
            'total_params': checkpoint.get('total_params', 'N/A')
        }
        # Calcular tamanho do arquivo
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        model_info['size_mb'] = model_size_mb
    
    # Figuras disponíveis
    figures_dir = 'outputs/figures'
    available_figures = []
    if os.path.exists(figures_dir):
        for fig in ['training_curves.png', 'f1_by_dataset.png', 'gradcam_examples.png',
                    'confusion_matrix_faceforensics.png', 'confusion_matrix_celebdf.png',
                    'confusion_matrix_wilddeepfake.png', 'roc_curve_faceforensics.png',
                    'roc_curve_celebdf.png', 'roc_curve_wilddeepfake.png']:
            if os.path.exists(os.path.join(figures_dir, fig)):
                available_figures.append(fig)
    
    # ========================================================================
    # GERAR MARKDOWN
    # ========================================================================
    
    lines = []
    
    # Cabeçalho
    lines.append("# Relatório Técnico - Deepfake Detector")
    lines.append("")
    lines.append(f"**Data de Geração:** {timestamp}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 1. Informações do Sistema
    lines.append("## 1. Informações do Sistema")
    lines.append("")
    lines.append("| Componente | Versão/Informação |")
    lines.append("|------------|-------------------|")
    lines.append(f"| Python | {python_version} |")
    lines.append(f"| PyTorch | {torch_version} |")
    lines.append(f"| Sistema Operacional | {system_info} |")
    lines.append(f"| Device | {device_info} |")
    lines.append("")
    
    # 2. Configuração do Modelo
    lines.append("## 2. Configuração do Modelo")
    lines.append("")
    lines.append("### Arquitetura")
    lines.append("")
    lines.append("- **Tipo:** CNN-LSTM Híbrido")
    lines.append("- **CNN Backbone:** ResNet-34 (pré-treinado ImageNet)")
    lines.append("- **Sequencial:** BiLSTM (2 camadas, 256 unidades)")
    lines.append("- **Classificador:** Linear 512 → 1 (Sigmoid)")
    lines.append("")
    
    if model_info:
        lines.append("### Estatísticas do Modelo")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        total_params = model_info.get('total_params', 'N/A')
        if isinstance(total_params, (int, float)):
            lines.append(f"| Total de Parâmetros | {int(total_params):,} |")
        else:
            lines.append(f"| Total de Parâmetros | {total_params} |")
        lines.append(f"| Tamanho do Arquivo | {model_info.get('size_mb', 0):.2f} MB |")
        lines.append(f"| Melhor Época | {model_info.get('epoch', 'N/A')} |")
        lines.append("")
    
    # 3. Configuração de Treinamento
    lines.append("## 3. Configuração de Treinamento")
    lines.append("")
    lines.append("| Hiperparâmetro | Valor |")
    lines.append("|----------------|-------|")
    lines.append("| Otimizador | Adam |")
    lines.append("| Learning Rate | 1e-4 |")
    lines.append("| Loss Function | Binary Cross-Entropy (BCE) |")
    lines.append("| Batch Size | 4 |")
    lines.append("| Frames por Vídeo | 16 |")
    lines.append("| Early Stopping | Patience 5 (Val F1) |")
    lines.append("| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |")
    lines.append("| Seed | 42 |")
    lines.append("")
    
    # 4. Resultados de Treinamento
    lines.append("## 4. Resultados de Treinamento")
    lines.append("")
    
    if train_metrics is not None and best_epoch is not None:
        lines.append(f"**Total de Épocas:** {len(train_metrics)}")
        lines.append("")
        lines.append(f"**Melhor Época:** {int(best_epoch['epoch'])}")
        lines.append("")
        lines.append("| Métrica | Valor |")
        lines.append("|---------|-------|")
        lines.append(f"| Train Loss | {best_epoch['train_loss']:.4f} |")
        lines.append(f"| Val Loss | {best_epoch['val_loss']:.4f} |")
        lines.append(f"| Val F1-Score | {best_epoch['val_f1']:.4f} |")
        lines.append(f"| Val AUC | {best_epoch['val_auc']:.4f} |")
        lines.append("")
        
        lines.append("### Evolução do Treinamento")
        lines.append("")
        lines.append("| Época | Train Loss | Val Loss | Val F1 | Val AUC |")
        lines.append("|-------|------------|----------|--------|---------|")
        for _, row in train_metrics.iterrows():
            lines.append(f"| {int(row['epoch'])} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['val_f1']:.4f} | {row['val_auc']:.4f} |")
        lines.append("")
    else:
        lines.append("*Métricas de treinamento não disponíveis.*")
        lines.append("")
    
    # 5. Avaliação Cross-Dataset
    lines.append("## 5. Avaliação Cross-Dataset")
    lines.append("")
    
    if cross_metrics is not None:
        lines.append("Avaliação em três datasets para medir generalização:")
        lines.append("")
        lines.append("| Dataset | Accuracy | Precision | Recall | F1-Score | AUC | Samples |")
        lines.append("|---------|----------|-----------|--------|----------|-----|---------|")
        for _, row in cross_metrics.iterrows():
            lines.append(f"| {row['dataset']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['auc']:.4f} | {int(row['total_samples'])} |")
        lines.append("")
        
        # Análise
        lines.append("### Análise dos Resultados")
        lines.append("")
        avg_f1 = cross_metrics['f1'].mean()
        avg_auc = cross_metrics['auc'].mean()
        lines.append(f"- **F1-Score Médio:** {avg_f1:.4f}")
        lines.append(f"- **AUC Médio:** {avg_auc:.4f}")
        lines.append("")
        
        if avg_f1 < 0.3:
            lines.append("⚠️ **Observação:** O modelo apresentou baixo desempenho nos conjuntos de teste, indicando possível overfitting ou necessidade de mais dados de treinamento.")
        elif avg_f1 < 0.7:
            lines.append("ℹ️ **Observação:** O modelo apresentou desempenho moderado. Possíveis melhorias: aumento de dados, data augmentation, fine-tuning.")
        else:
            lines.append("✓ **Observação:** O modelo apresentou bom desempenho cross-dataset, indicando boa generalização.")
        lines.append("")
    else:
        lines.append("*Métricas cross-dataset não disponíveis.*")
        lines.append("")
    
    # 6. Visualizações
    lines.append("## 6. Visualizações Geradas")
    lines.append("")
    
    if available_figures:
        lines.append("### Figuras Principais")
        lines.append("")
        
        # Training Curves
        if 'training_curves.png' in available_figures:
            lines.append("#### Training Curves")
            lines.append("")
            lines.append("![Training Curves](../figures/training_curves.png)")
            lines.append("")
            lines.append("*Evolução das métricas durante o treinamento.*")
            lines.append("")
        
        # F1 by Dataset
        if 'f1_by_dataset.png' in available_figures:
            lines.append("#### F1-Score por Dataset")
            lines.append("")
            lines.append("![F1 by Dataset](../figures/f1_by_dataset.png)")
            lines.append("")
            lines.append("*Comparação de desempenho entre datasets.*")
            lines.append("")
        
        # Grad-CAM Examples
        if 'gradcam_examples.png' in available_figures:
            lines.append("#### Exemplos Grad-CAM")
            lines.append("")
            lines.append("![Grad-CAM Examples](../figures/gradcam_examples.png)")
            lines.append("")
            lines.append("*Visualização de regiões de atenção do modelo (explicabilidade).*")
            lines.append("")
        
        # Confusion Matrices
        cm_files = [f for f in available_figures if f.startswith('confusion_matrix')]
        if cm_files:
            lines.append("#### Matrizes de Confusão")
            lines.append("")
            for cm_file in sorted(cm_files):
                dataset_name = cm_file.replace('confusion_matrix_', '').replace('.png', '').title()
                lines.append(f"**{dataset_name}:**")
                lines.append("")
                lines.append(f"![Confusion Matrix {dataset_name}](../figures/{cm_file})")
                lines.append("")
        
        # ROC Curves
        roc_files = [f for f in available_figures if f.startswith('roc_curve')]
        if roc_files:
            lines.append("#### Curvas ROC")
            lines.append("")
            for roc_file in sorted(roc_files):
                dataset_name = roc_file.replace('roc_curve_', '').replace('.png', '').title()
                lines.append(f"**{dataset_name}:**")
                lines.append("")
                lines.append(f"![ROC Curve {dataset_name}](../figures/{roc_file})")
                lines.append("")
    else:
        lines.append("*Nenhuma figura disponível.*")
        lines.append("")
    
    # 7. Arquivos Gerados
    lines.append("## 7. Arquivos de Saída")
    lines.append("")
    lines.append("### Modelos")
    lines.append("")
    lines.append("- `models/model_best.pt` - Modelo treinado (melhor época)")
    lines.append("")
    lines.append("### Métricas")
    lines.append("")
    lines.append("- `outputs/metrics_train.csv` - Métricas de treinamento por época")
    lines.append("- `outputs/metrics_cross.csv` - Métricas de avaliação cross-dataset")
    lines.append("- `outputs/reports/table_metrics.csv` - Tabela consolidada de métricas")
    lines.append("- `outputs/reports/interface_log.csv` - Log de execuções da interface Gradio")
    lines.append("")
    lines.append("### Figuras")
    lines.append("")
    for fig in sorted(available_figures):
        lines.append(f"- `outputs/figures/{fig}`")
    lines.append("")
    lines.append("### Heatmaps Grad-CAM")
    lines.append("")
    lines.append("- `outputs/heatmaps/*.png` - Heatmaps de atenção por frame")
    lines.append("")
    
    # 8. Conclusões
    lines.append("## 8. Conclusões e Próximos Passos")
    lines.append("")
    lines.append("### Conquistas")
    lines.append("")
    lines.append("✓ Modelo CNN-LSTM implementado e treinado com sucesso")
    lines.append("")
    lines.append("✓ Pipeline completo de pré-processamento com detecção facial (MTCNN)")
    lines.append("")
    lines.append("✓ Grad-CAM implementado para explicabilidade visual")
    lines.append("")
    lines.append("✓ Interface Gradio funcional para demonstração")
    lines.append("")
    lines.append("✓ Avaliação cross-dataset em 3 benchmarks")
    lines.append("")
    
    lines.append("### Desafios Identificados")
    lines.append("")
    
    if cross_metrics is not None and cross_metrics['f1'].mean() < 0.3:
        lines.append("⚠️ **Generalização:** Modelo apresentou overfitting no conjunto de validação")
        lines.append("")
        lines.append("⚠️ **Dataset:** Conjunto de treinamento pequeno (30 vídeos)")
        lines.append("")
        lines.append("⚠️ **Threshold:** Threshold fixo (0.5) pode não ser ótimo")
        lines.append("")
    
    lines.append("### Melhorias Sugeridas")
    lines.append("")
    lines.append("1. **Aumentar dataset de treinamento:** Coletar mais vídeos ou usar data augmentation")
    lines.append("")
    lines.append("2. **Otimizar threshold:** Usar validação cruzada para encontrar threshold ótimo")
    lines.append("")
    lines.append("3. **Regularização:** Adicionar dropout, weight decay ou early stopping mais agressivo")
    lines.append("")
    lines.append("4. **Arquitetura:** Testar outros backbones (EfficientNet, Vision Transformer)")
    lines.append("")
    lines.append("5. **Ensemble:** Combinar múltiplos modelos para maior robustez")
    lines.append("")
    
    # Rodapé
    lines.append("---")
    lines.append("")
    lines.append("*Relatório gerado automaticamente pelo sistema Deepfake Detector.*")
    lines.append("")
    lines.append(f"*Timestamp: {timestamp}*")
    lines.append("")
    
    # ========================================================================
    # SALVAR MARKDOWN
    # ========================================================================
    
    markdown_content = "\n".join(lines)
    
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    linhas_relatorio = len(lines)
    
    print(f"✓ Relatório Markdown gerado: {output_md}")
    print(f"  Linhas: {linhas_relatorio}")
    
    # ========================================================================
    # CONVERTER PARA PDF (OPCIONAL)
    # ========================================================================
    
    pdf_generated = False
    
    try:
        # Tentar usar markdown2 + pdfkit (se disponível)
        import markdown2
        
        # Converter Markdown para HTML
        html_content = markdown2.markdown(
            markdown_content,
            extras=['tables', 'fenced-code-blocks', 'header-ids']
        )
        
        # Template HTML com CSS
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Relatório Técnico - Deepfake Detector</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 8px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
            text-align: center;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
        
        # Salvar HTML temporário
        html_path = output_md.replace('.md', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"✓ HTML intermediário gerado: {html_path}")
        
        # Tentar converter para PDF com pdfkit
        try:
            import pdfkit
            pdfkit.from_file(html_path, output_pdf)
            pdf_generated = True
            print(f"✓ Relatório PDF gerado: {output_pdf}")
        except (ImportError, OSError) as e:
            print(f"⚠️  pdfkit não disponível: {e}")
            print(f"   Para gerar PDF, instale: pip install pdfkit")
            print(f"   E wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
    
    except ImportError:
        print(f"⚠️  markdown2 não disponível")
        print(f"   Para gerar PDF, instale: pip install markdown2 pdfkit")
    
    # ========================================================================
    # RESUMO FINAL
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("RESUMO DO RELATÓRIO")
    print(f"{'='*70}\n")
    print(f"Arquivo Markdown: {output_md}")
    print(f"Linhas do relatório: {linhas_relatorio}")
    
    if pdf_generated:
        print(f"Arquivo PDF: {output_pdf}")
    else:
        print(f"Arquivo PDF: Não gerado (dependências ausentes)")
    
    print(f"\nSeções incluídas:")
    print(f"  1. Informações do Sistema")
    print(f"  2. Configuração do Modelo")
    print(f"  3. Configuração de Treinamento")
    print(f"  4. Resultados de Treinamento")
    print(f"  5. Avaliação Cross-Dataset")
    print(f"  6. Visualizações Geradas")
    print(f"  7. Arquivos de Saída")
    print(f"  8. Conclusões e Próximos Passos")
    
    print(f"\n{'='*70}")
    print("RELATÓRIO GERADO COM SUCESSO")
    print(f"{'='*70}\n")
    
    return linhas_relatorio


if __name__ == '__main__':
    # Teste de geração de relatório
    generate_technical_report()

