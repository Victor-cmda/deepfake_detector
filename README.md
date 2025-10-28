# Deepfake Detector

Protótipo de detecção de deepfakes com explicabilidade visual utilizando CNN-LSTM e Grad-CAM.

## 📋 Especificações Técnicas

- **Python**: 3.11.5
- **Framework**: PyTorch >= 2.2
- **Arquitetura**: ResNet-34 + BiLSTM (256x2 camadas)
- **Datasets**: FaceForensics++, Celeb-DF-v2, WildDeepfake
- **Explicabilidade**: Grad-CAM para visualização de atenção
- **Interface**: Gradio para demonstração interativa

## 📂 Estrutura do Project

```
deepfake_detector/
├── data/                           # Datasets e índices
│   ├── celebdf/                   # Celeb-DF-v2 dataset
│   │   ├── videos_real/          # Vídeos reais
│   │   └── videos_fake/          # Vídeos fake
│   ├── faceforensicspp/          # FaceForensics++ dataset
│   │   ├── videos_real/          # Vídeos reais
│   │   └── videos_fake/          # Vídeos fake
│   ├── wilddeepfake/             # WildDeepfake dataset
│   │   ├── videos_real/          # Vídeos reais
│   │   └── videos_fake/          # Vídeos fake
│   ├── celebdf_index.csv         # Índice Celeb-DF-v2
│   ├── faceforensicspp_index.csv # Índice FaceForensics++
│   ├── wilddeepfake_index.csv    # Índice WildDeepfake
│   └── splits_faceforensicspp.csv # Divisão treino/val/teste (70/15/15)
│
├── models/                         # Modelos treinados
│   └── model_best.pt              # Melhor modelo (Val F1=1.0)
│
├── outputs/                        # Resultados e artefatos
│   ├── figures/                   # Gráficos e visualizações
│   │   ├── training_curves.png           # Curvas de treinamento
│   │   ├── f1_by_dataset.png             # F1-Score por dataset
│   │   ├── gradcam_examples.png          # Exemplos Grad-CAM
│   │   ├── robustness.png                # Análise de robustez
│   │   ├── confusion_matrix_faceforensics.png
│   │   ├── confusion_matrix_celebdf.png
│   │   ├── confusion_matrix_wilddeepfake.png
│   │   ├── roc_curve_faceforensics.png
│   │   ├── roc_curve_celebdf.png
│   │   ├── roc_curve_wilddeepfake.png
│   │   ├── model_architecture.png
│   │   ├── preprocessing_example.png
│   │   └── preprocessing_comparison.png
│   │
│   ├── heatmaps/                  # Mapas de calor Grad-CAM
│   │   └── [vídeos com overlays de atenção]
│   │
│   ├── logs/                      # Logs de execução
│   │   ├── early_stopping.txt    # Log de early stopping
│   │   ├── preprocessing_stats.txt
│   │   ├── model_specs.txt
│   │   ├── dataloader_stats.txt
│   │   └── step_*.txt            # Logs detalhados das tarefas 1-14
│   │
│   ├── reports/                   # Relatórios e métricas
│   │   ├── table_metrics.csv     # Consolidação de métricas
│   │   ├── robustness.csv        # Testes de robustez
│   │   ├── interface_log.csv     # Log da interface Gradio
│   │   └── run_report.md         # Relatório técnico completo
│   │
│   ├── metrics_train.csv          # Métricas de treinamento
│   └── metrics_cross.csv          # Métricas cross-dataset
│
├── src/                            # Código fonte
│   ├── preprocessing.py           # Extração de frames, detecção facial, normalização
│   ├── model.py                   # Arquitetura DeepfakeDetector
│   ├── gradcam.py                 # Implementação Grad-CAM
│   ├── train.py                   # Pipeline de treinamento
│   ├── evaluate.py                # Avaliação e robustez
│   ├── interface.py               # Interface Gradio
│   └── utils.py                   # Utilidades e relatórios
│
├── main.py                         # Script principal (não usado)
├── requirements.txt                # Dependências Python
├── instructions.json               # Especificações do projeto
└── README.md                       # Este arquivo
```

## 🚀 Instalação

### Pré-requisitos

- **Python 3.11.5**: Versão exata requerida
- **CUDA** (opcional): Para treinamento em GPU NVIDIA
- **MPS** (macOS): Suporte para Apple Silicon

### Passo 1: Clonar o repositório

```bash
git clone <repository-url>
cd deepfake_detector
```

### Passo 2: Configurar Python 3.11.5

**Usando pyenv (recomendado):**

```bash
# Instalar pyenv (se necessário)
curl https://pyenv.run | bash

# Instalar Python 3.11.5
pyenv install 3.11.5

# Configurar versão local
pyenv local 3.11.5

# Verificar versão
python --version  # Deve exibir: Python 3.11.5
```

**Usando conda:**

```bash
conda create -n deepfake python=3.11.5
conda activate deepfake
```

### Passo 3: Instalar dependências

```bash
pip install -r requirements.txt
```

**Dependências principais:**
- `torch>=2.2.0` - Framework de deep learning
- `torchvision>=0.17.0` - Utilitários para visão computacional
- `facenet-pytorch` - Detecção facial (MTCNN)
- `opencv-python` - Processamento de vídeo
- `gradio` - Interface web interativa
- `matplotlib`, `seaborn` - Visualização
- `pandas`, `numpy` - Manipulação de dados
- `scikit-learn` - Métricas de avaliação
- `tqdm` - Barras de progresso

### Passo 4: Organizar datasets

**Estrutura esperada:**

```bash
deepfake_detector/
└── data/
    ├── faceforensicspp/
    │   ├── videos_real/
    │   │   ├── video_001.mp4
    │   │   └── ...
    │   └── videos_fake/
    │       ├── video_001.mp4
    │       └── ...
    ├── celebdf/
    │   ├── videos_real/
    │   └── videos_fake/
    └── wilddeepfake/
        ├── videos_real/
        └── videos_fake/
```

**Gerar índices dos datasets:**

```bash
# Executar a tarefa 2 para criar os índices CSV
python -c "from src.preprocessing import create_dataset_index; \
create_dataset_index('data/faceforensicspp', 'data/faceforensicspp_index.csv'); \
create_dataset_index('data/celebdf', 'data/celebdf_index.csv'); \
create_dataset_index('data/wilddeepfake', 'data/wilddeepfake_index.csv')"
```

**Gerar divisão treino/validação/teste:**

```bash
# Executar a tarefa 3 para criar splits (70/15/15)
python -c "from src.preprocessing import create_train_val_test_split; \
create_train_val_test_split('data/faceforensicspp_index.csv', 'data/splits_faceforensicspp.csv')"
```

## 🎯 Treinamento

### Treinar o modelo

```bash
python src/train.py
```

**Configurações de treinamento:**
- **Otimizador**: Adam (lr=1e-4, weight_decay=1e-5)
- **Loss**: Binary Cross-Entropy (BCE)
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Early Stopping**: Patience=5 épocas
- **Batch Size**: 4 (ajustável conforme memória)
- **Frames por vídeo**: 16
- **Epochs**: Até convergência (early stopping)

**Outputs gerados:**
- `models/model_best.pt` - Melhor modelo salvo
- `outputs/metrics_train.csv` - Métricas por época
- `outputs/logs/early_stopping.txt` - Log de parada antecipada
- `outputs/figures/training_curves.png` - Curvas de loss e métricas

**Exemplo de execução:**

```
Epoch 1/100
Train Loss: 0.4234, Val Loss: 0.3156, Val F1: 0.8923
Melhor modelo salvo!

Epoch 2/100
Train Loss: 0.2891, Val Loss: 0.0234, Val F1: 1.0000
Melhor modelo salvo!

Early stopping acionado na época 7
Melhor Val F1: 1.0000 (época 2)
```

### Métricas de treinamento

O arquivo `outputs/metrics_train.csv` contém:
- `epoch` - Número da época
- `train_loss` - Loss no conjunto de treino
- `val_loss` - Loss no conjunto de validação
- `val_accuracy` - Acurácia de validação
- `val_precision` - Precisão de validação
- `val_recall` - Recall de validação
- `val_f1` - F1-Score de validação
- `val_auc` - AUC-ROC de validação
- `learning_rate` - Taxa de aprendizado atual

## 📊 Avaliação

### Avaliação cross-dataset

```bash
python src/evaluate.py
```

**Datasets avaliados:**
1. **FaceForensics++** (test split)
2. **Celeb-DF-v2** (completo)
3. **WildDeepfake** (completo)

**Outputs gerados:**
- `outputs/metrics_cross.csv` - Métricas por dataset
- `outputs/figures/confusion_matrix_*.png` - Matrizes de confusão
- `outputs/figures/roc_curve_*.png` - Curvas ROC
- `outputs/figures/f1_by_dataset.png` - Comparação F1-Score
- `outputs/reports/table_metrics.csv` - Tabela consolidada

**Métricas calculadas:**
- Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion Matrix
- True Positives, False Positives, True Negatives, False Negatives

### Gerar Grad-CAM

```bash
python src/gradcam.py
```

**Configurações:**
- **Layer alvo**: `model.resnet.layer4`
- **Frames por vídeo**: 8 (igualmente espaçados)
- **Output**: Heatmaps em `outputs/heatmaps/`

**Exemplo de uso:**

```python
from src.gradcam import generate_gradcam

# Gerar Grad-CAM para um vídeo
heatmaps = generate_gradcam(
    model_path='models/model_best.pt',
    video_path='data/celebdf/videos_fake/video_001.mp4',
    output_dir='outputs/heatmaps/',
    num_frames=8
)
```

**Outputs:**
- `outputs/heatmaps/<video_name>_gradcam.mp4` - Vídeo com overlay
- `outputs/figures/gradcam_examples.png` - Grid 2x3 de exemplos

### Teste de robustez

```bash
# Executar teste de robustez (já incluído no evaluate.py)
python src/evaluate.py
```

**Degradações testadas:**
1. **Original** (baseline)
2. **Ruído Gaussiano**: σ ∈ {0.01, 0.05, 0.10}
3. **Blur Gaussiano**: kernel ∈ {3, 7, 15}
4. **Compressão JPEG**: quality ∈ {90, 50, 20}
5. **Redimensionamento**: scale ∈ {75%, 50%, 25%}

**Outputs:**
- `outputs/reports/robustness.csv` - Resultados detalhados
- `outputs/figures/robustness.png` - Gráfico de barras

**Métricas:**
- `delta_probabilidade` - Mudança absoluta na probabilidade
- Estatísticas: média, máximo, mínimo, desvio padrão

**Exemplo de resultados:**

```
Degradação mais impactante: Blur k=15 (Δ = 0.0274)
Δ médio: 0.0110 (1.1%)
Modelo MUITO ROBUSTO
```

## 🖥️ Interface Gradio

### Iniciar interface web

```bash
python src/interface.py
```

**URL local:** http://127.0.0.1:7860

**Funcionalidades:**
- Upload de vídeo (.mp4, .avi, .mov)
- Detecção automática de deepfake
- Visualização Grad-CAM
- Probabilidade de fake (0-100%)
- Log de execuções em `outputs/reports/interface_log.csv`

**Exemplo de uso programático:**

```python
from src.interface import predict

# Fazer predição
result = predict('data/celebdf/videos_fake/video_001.mp4')

print(f"Probabilidade Fake: {result['probability']:.2%}")
print(f"Label: {result['label']}")
print(f"Tempo: {result['inference_time']:.2f}s")
```

**Interface log:**

O arquivo `outputs/reports/interface_log.csv` registra:
- `timestamp` - Data/hora da execução
- `video_path` - Caminho do vídeo
- `probabilidade_fake` - Probabilidade predita
- `label` - REAL ou FAKE
- `tempo_inferencia` - Tempo em segundos

## 📈 Relatórios

### Gerar relatório técnico completo

```bash
python -c "from src.utils import generate_technical_report; generate_technical_report()"
```

**Output:** `outputs/reports/run_report.md`

**Seções do relatório:**
1. Informações do Sistema
2. Configurações do Modelo
3. Métricas de Treinamento
4. Métricas Cross-Dataset
5. Análise de Robustez
6. Grad-CAM e Explicabilidade
7. Logs de Interface
8. Arquivos Gerados

### Gerar todas as figuras

```bash
python -c "from src.evaluate import generate_all_figures_and_reports; generate_all_figures_and_reports()"
```

**Figuras geradas:**
- `training_curves.png` - Loss e métricas (4200x1500px, 300 DPI)
- `f1_by_dataset.png` - F1-Score comparativo (3000x1800px, 300 DPI)
- `gradcam_examples.png` - Grid 2x3 de exemplos (2250x1500px, 150 DPI)

## 📋 Arquivos Finais

### Modelos

- `models/model_best.pt` (93.4 MB) - Modelo treinado

### Métricas

- `outputs/metrics_train.csv` - Histórico de treinamento
- `outputs/metrics_cross.csv` - Avaliação cross-dataset
- `outputs/reports/table_metrics.csv` - Tabela consolidada
- `outputs/reports/robustness.csv` - Testes de robustez
- `outputs/reports/interface_log.csv` - Log da interface

### Figuras

- `outputs/figures/training_curves.png` - Curvas de treinamento
- `outputs/figures/f1_by_dataset.png` - F1-Score por dataset
- `outputs/figures/gradcam_examples.png` - Exemplos Grad-CAM
- `outputs/figures/robustness.png` - Análise de robustez
- `outputs/figures/confusion_matrix_*.png` - Matrizes de confusão (3)
- `outputs/figures/roc_curve_*.png` - Curvas ROC (3)
- `outputs/figures/model_architecture.png` - Arquitetura do modelo
- `outputs/figures/preprocessing_*.png` - Exemplos de pré-processamento (2)

### Relatórios

- `outputs/reports/run_report.md` - Relatório técnico completo

### Logs

- `outputs/logs/early_stopping.txt` - Log de parada antecipada
- `outputs/logs/preprocessing_stats.txt` - Estatísticas de pré-processamento
- `outputs/logs/model_specs.txt` - Especificações do modelo
- `outputs/logs/dataloader_stats.txt` - Estatísticas do DataLoader
- `outputs/logs/step_*.txt` - Logs detalhados das tarefas 1-14

## 🔧 Comandos Úteis

### Pipeline completo

```bash
# 1. Criar índices dos datasets
python -c "from src.preprocessing import create_dataset_index; \
create_dataset_index('data/faceforensicspp', 'data/faceforensicspp_index.csv'); \
create_dataset_index('data/celebdf', 'data/celebdf_index.csv'); \
create_dataset_index('data/wilddeepfake', 'data/wilddeepfake_index.csv')"

# 2. Criar divisão treino/val/teste
python -c "from src.preprocessing import create_train_val_test_split; \
create_train_val_test_split('data/faceforensicspp_index.csv', 'data/splits_faceforensicspp.csv')"

# 3. Treinar modelo
python src/train.py

# 4. Avaliar modelo (cross-dataset + robustez)
python src/evaluate.py

# 5. Gerar figuras e relatórios
python -c "from src.evaluate import generate_all_figures_and_reports; generate_all_figures_and_reports()"

# 6. Gerar relatório técnico
python -c "from src.utils import generate_technical_report; generate_technical_report()"

# 7. Iniciar interface Gradio
python src/interface.py
```

### Verificar estrutura

```bash
# Listar arquivos gerados
ls -lh models/
ls -lh outputs/metrics_*.csv
ls -lh outputs/figures/
ls -lh outputs/reports/
ls -lh outputs/logs/

# Verificar modelo treinado
python -c "import torch; ckpt = torch.load('models/model_best.pt', map_location='cpu'); \
print(f'Epoch: {ckpt[\"epoch\"]}'); print(f'Val F1: {ckpt[\"val_f1\"]:.4f}')"

# Ver métricas de treinamento
head -10 outputs/metrics_train.csv

# Ver métricas cross-dataset
cat outputs/metrics_cross.csv
```

### Predição em lote

```bash
# Avaliar todos os vídeos de um diretório
python -c "
import os
from src.interface import predict
from pathlib import Path

video_dir = 'data/celebdf/videos_fake'
for video_file in Path(video_dir).glob('*.mp4'):
    result = predict(str(video_file))
    print(f'{video_file.name}: {result[\"probability\"]:.2%} ({result[\"label\"]})')
"
```

### Visualizar Grad-CAM específico

```bash
python -c "
from src.gradcam import generate_gradcam

generate_gradcam(
    model_path='models/model_best.pt',
    video_path='data/celebdf/videos_fake/video_001.mp4',
    output_dir='outputs/heatmaps/',
    num_frames=8
)
"
```

## 📊 Resultados Esperados

### Métricas de Treinamento

- **Val F1-Score**: 1.0000 (época 2)
- **Val AUC-ROC**: ~0.99+
- **Early Stopping**: Acionado em ~7 épocas
- **Tempo por época**: ~5-10 min (GPU) ou ~30-60 min (CPU)

### Métricas Cross-Dataset

| Dataset          | Accuracy | Precision | Recall | F1-Score | AUC   |
|------------------|----------|-----------|--------|----------|-------|
| FaceForensics++  | ~0.98+   | ~0.95+    | ~0.95+ | ~0.95+   | ~0.99 |
| Celeb-DF-v2      | ~0.85+   | ~0.80+    | ~0.85+ | ~0.82+   | ~0.90 |
| WildDeepfake     | ~0.75+   | ~0.70+    | ~0.75+ | ~0.72+   | ~0.85 |

### Robustez

- **Δ probabilidade médio**: 0.0110 (1.1%)
- **Degradação mais impactante**: Blur k=15 (Δ=0.027)
- **Modelo**: MUITO ROBUSTO

## 🐛 Troubleshooting

### Erro: CUDA out of memory

```bash
# Reduzir batch size em src/train.py
# Linha: batch_size = 4  →  batch_size = 2 ou 1
```

### Erro: MTCNN não detecta faces

```bash
# Verificar qualidade dos vídeos
# MTCNN requer rostos visíveis e bem iluminados
# Ajustar threshold em src/preprocessing.py se necessário
```

### Erro: Módulo não encontrado

```bash
# Reinstalar dependências
pip install -r requirements.txt --force-reinstall
```

### Interface Gradio não abre

```bash
# Verificar se a porta 7860 está disponível
# Ou especificar porta diferente:
# Em src/interface.py, modificar: demo.launch(server_port=7861)
```

## 📄 Licença

Projeto desenvolvido como parte do TCC - Capítulo 4.

## 👤 Autor

Desenvolvido para detecção de deepfakes com explicabilidade visual utilizando arquitetura CNN-LSTM e Grad-CAM.

---

**Data de última atualização**: 28 de outubro de 2025
