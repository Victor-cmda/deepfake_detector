# Relatório Técnico - Deepfake Detector

**Data de Geração**: 01/11/2025 18:31:33

## 📊 Resumo Executivo

Este relatório apresenta os resultados do sistema de detecção de deepfakes
desenvolvido como parte do TCC, utilizando arquitetura CNN-LSTM com explicabilidade visual.

## 🎯 Objetivos Alcançados

- ✅ Treinamento completo do modelo (20 épocas)
- ✅ Cross-dataset evaluation (múltiplos datasets)
- ✅ Implementação de Grad-CAM para interpretabilidade
- ✅ Interface web funcional com Gradio

## 📈 Métricas Principais

### Treinamento

- **Melhor Época**: 17
- **Val AUC**: 0.8507
- **Val F1-Score**: 0.9269
- **Val Loss**: 0.5274

### Cross-Dataset Evaluation

#### FaceForensics++

- **AUC**: 0.8370
- **F1-Score**: 0.9287
- **Accuracy**: 0.8743
- **Precision**: 0.9034
- **Recall**: 0.9556
- **Amostras Testadas**: 1050

#### Celeb-DF-v2

- **AUC**: 0.7309
- **F1-Score**: 0.9291
- **Accuracy**: 0.8698
- **Precision**: 0.8768
- **Recall**: 0.9881
- **Amostras Testadas**: 6529

## 📁 Figuras Geradas

Todas as visualizações estão disponíveis em `outputs/figures/`:

- `training_curves.png` - Curvas de treinamento (loss, AUC, F1)
- `f1_by_dataset.png` - Comparação de F1-Score entre datasets
- `confusion_matrix.png` - Matrizes de confusão
- `gradcam_examples.png` - Exemplos de mapas de atenção Grad-CAM

## 🔬 Especificações Técnicas

- **Arquitetura**: ResNet-34 + BiLSTM (24.4M parâmetros)
- **Framework**: PyTorch 2.5.1 + CUDA 12.1
- **Hardware**: NVIDIA GeForce RTX 4060 (8GB)
- **Datasets**: FaceForensics++ (7.000 vídeos) + Celeb-DF-v2 (6.529 vídeos)

## 📝 Conclusão

O sistema demonstrou capacidade robusta de detecção de deepfakes,
com AUC superior a 74% em cross-dataset evaluation e interpretabilidade
visual através de Grad-CAM.

---
*Relatório gerado automaticamente em 01/11/2025*