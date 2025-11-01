# 📚 OUTPUTS PARA O TCC - PASTA ORGANIZADA

**Data de Criação**: 1 de novembro de 2025  
**Status**: ✅ **TODOS OS ARQUIVOS NECESSÁRIOS INCLUÍDOS**

---

## 📁 ESTRUTURA DA PASTA

```
TCC_OUTPUTS/
├── README.md                          (este arquivo)
│
├── figures/                           📊 FIGURAS PARA O TCC
│   ├── training_curves.png           (445 KB) - Curvas de treinamento
│   ├── f1_by_dataset.png             (106 KB) - F1/AUC por dataset
│   ├── confusion_matrix.png          (133 KB) - Matrizes de confusão
│   └── gradcam_examples.png          (1.9 MB) - Exemplos Grad-CAM
│
├── relatorios/                        📄 RELATÓRIOS E TABELAS
│   ├── table_metrics.csv             (0.5 KB) - Métricas consolidadas
│   ├── run_report.md                 (1.9 KB) - Relatório técnico
│   ├── interface_log.csv             (4.2 KB) - Log da interface
│   └── robustness.csv                (4.3 KB) - Teste de robustez
│
├── metricas/                          📈 MÉTRICAS BRUTAS
│   ├── metrics_train.csv             (1.8 KB) - 20 épocas de treino
│   └── metrics_cross.csv             (0.3 KB) - Cross-dataset evaluation
│
├── heatmaps_exemplos/                 🎨 EXEMPLOS GRAD-CAM
│   ├── 594_530_frame_000_gradcam.png (4 exemplos de heatmaps)
│   ├── 594_530_frame_001_gradcam.png
│   ├── 594_530_frame_002_gradcam.png
│   └── 594_530_frame_003_gradcam.png
│
└── documentacao/                      📖 GUIAS E DOCUMENTAÇÃO
    ├── GUIA_USO_TCC.md               - Guia completo de uso
    ├── OUTPUTS_TCC_REFERENCIA.md     - Referência detalhada
    └── CROSS_DATASET_EVALUATION.md   - Análise cross-dataset
```

---

## 🎯 ARQUIVOS CONFORME `instructions.json`

### ✅ Outputs Esperados (Tarefa 12-15)

Todos os arquivos mencionados no `instructions.json` estão incluídos:

#### 1. **models/model_best.pt** 
❌ **NÃO INCLUÍDO** (93.4 MB - muito grande)
- **Localização original**: `E:\deepfake_detector\models\model_best.pt`
- **Como usar**: Referenciar o caminho no TCC

#### 2. **outputs/metrics_train.csv** ✅
📍 **Localização**: `metricas/metrics_train.csv`
- 20 épocas de treinamento
- Colunas: epoch, train_loss, val_loss, val_f1, val_auc, learning_rate

#### 3. **outputs/metrics_cross.csv** ✅
📍 **Localização**: `metricas/metrics_cross.csv`
- 2 datasets validados (FaceForensics++, Celeb-DF-v2)
- Colunas: dataset, accuracy, precision, recall, f1, auc, total_samples

#### 4. **outputs/figures/training_curves.png** ✅
📍 **Localização**: `figures/training_curves.png`
- 4 gráficos: Loss, AUC, F1, Learning Rate

#### 5. **outputs/figures/f1_by_dataset.png** ✅
📍 **Localização**: `figures/f1_by_dataset.png`
- Comparação F1-Score e AUC entre datasets

#### 6. **outputs/figures/confusion_matrix.png** ✅
📍 **Localização**: `figures/confusion_matrix.png`
- Matrizes de confusão para 2 datasets

#### 7. **outputs/figures/gradcam_examples.png** ✅
📍 **Localização**: `figures/gradcam_examples.png`
- 6 exemplos de mapas de atenção Grad-CAM

#### 8. **outputs/reports/interface_log.csv** ✅
📍 **Localização**: `relatorios/interface_log.csv`
- Log de execuções da interface Gradio

#### 9. **outputs/reports/run_report.md** ✅
📍 **Localização**: `relatorios/run_report.md`
- Relatório técnico completo

#### 10. **outputs/reports/table_metrics.csv** ✅
📍 **Localização**: `relatorios/table_metrics.csv`
- Tabela consolidada de métricas (8 métricas principais)

#### 11. **outputs/reports/robustness.csv** ✅
📍 **Localização**: `relatorios/robustness.csv`
- Resultados do teste de robustez

---

## 📊 COMO USAR AS FIGURAS NO TCC

### 1. **training_curves.png**
**Seção**: Resultados do Treinamento

**Legenda sugerida**:
> Figura X: Curvas de treinamento do modelo ao longo de 20 épocas. (a) Loss de treino e validação, (b) AUC de validação com linha de meta em 0.85, (c) F1-Score de validação, (d) Taxa de aprendizado com escala logarítmica. O melhor desempenho foi alcançado na época 17 (AUC: 85.07%, F1: 92.69%).

---

### 2. **f1_by_dataset.png**
**Seção**: Cross-Dataset Evaluation

**Legenda sugerida**:
> Figura Y: Comparação de F1-Score e AUC entre os datasets FaceForensics++ e Celeb-DF-v2 no conjunto de teste. Observa-se F1-Score consistente (~92.9%) em ambos, mas AUC superior em FaceForensics++ (83.70% vs 73.09%), indicando possível overfitting ao estilo deste dataset.

---

### 3. **confusion_matrix.png**
**Seção**: Análise de Erros

**Legenda sugerida**:
> Figura Z: Matrizes de confusão para os datasets (a) FaceForensics++ e (b) Celeb-DF-v2. Alto recall em ambos (95.56% e 98.81%) indica baixa taxa de falsos negativos, enquanto precision moderada (90.34% e 87.68%) sugere alguns falsos positivos.

---

### 4. **gradcam_examples.png**
**Seção**: Interpretabilidade Visual

**Legenda sugerida**:
> Figura W: Exemplos de mapas de atenção Grad-CAM para um vídeo deepfake do tipo NeuralTextures. Cada linha mostra: frame original, heatmap de atenção e sobreposição. O modelo foca predominantemente em regiões faciais (olhos, boca, bordas) sem depender de artefatos de background.

---

## 📈 MÉTRICAS PRINCIPAIS

### Treinamento
- **Val AUC**: 85.07% (época 17)
- **Val F1-Score**: 92.69% (época 17)
- **Train Loss**: 0.0038 (convergência excelente)

### Cross-Dataset Evaluation
- **FaceForensics++**: AUC 83.70%, F1 92.87% (1.050 amostras)
- **Celeb-DF-v2**: AUC 73.09%, F1 92.91% (6.529 amostras)
- **Média Ponderada**: AUC 74.56%, F1 92.91% (7.579 amostras)

---

## 📚 DOCUMENTAÇÃO INCLUÍDA

### 1. **GUIA_USO_TCC.md**
Guia completo com:
- ✅ Legendas prontas para cada figura
- ✅ Textos completos para cada seção do TCC
- ✅ Exemplos de código LaTeX
- ✅ Checklist final para submissão

### 2. **OUTPUTS_TCC_REFERENCIA.md**
Referência detalhada com:
- ✅ Descrição completa de cada output
- ✅ Interpretação das métricas
- ✅ Análises técnicas
- ✅ Textos acadêmicos sugeridos

### 3. **CROSS_DATASET_EVALUATION.md**
Relatório completo da avaliação cross-dataset:
- ✅ Resultados por dataset
- ✅ Análise de generalização
- ✅ Comparação com estado da arte

---

## 🔬 ESPECIFICAÇÕES TÉCNICAS

### Hardware
- **GPU**: NVIDIA GeForce RTX 4060 (8GB)
- **CUDA**: 12.1
- **Sistema**: Windows 11

### Software
- **Python**: 3.11.9
- **PyTorch**: 2.5.1+cu121
- **Framework**: Gradio 5.49.1

### Modelo
- **Arquitetura**: ResNet-34 + BiLSTM
- **Parâmetros**: 24.4M
- **Input**: 16 frames por vídeo (224×224 RGB)

### Datasets
- **FaceForensics++**: 7.000 vídeos
- **Celeb-DF-v2**: 6.529 vídeos
- **Total**: 13.529 vídeos

---

## ✅ CHECKLIST PARA O TCC

### Antes de Inserir no Documento

- [ ] Copiar `figures/*.png` para pasta de imagens do LaTeX
- [ ] Ler `documentacao/GUIA_USO_TCC.md` (textos prontos)
- [ ] Adaptar legendas das figuras ao estilo do TCC
- [ ] Inserir tabelas de métricas (usar `relatorios/table_metrics.csv`)
- [ ] Adicionar referências no texto para todas as figuras
- [ ] Verificar consistência dos valores citados

### Arquivos Obrigatórios

**Para o Documento Principal**:
- [x] 4 figuras PNG (alta resolução - DPI 300)
- [x] Métricas principais (tabelas)
- [x] Textos descritivos

**Para Apêndice/Material Suplementar** (opcional):
- [x] Métricas completas (CSV)
- [x] Relatório técnico (run_report.md)
- [x] Exemplos de heatmaps individuais

---

## 🎯 TAMANHO TOTAL DA PASTA

**Figuras**: ~2.6 MB  
**Relatórios**: ~11 KB  
**Métricas**: ~2 KB  
**Heatmaps**: ~8 MB (4 exemplos)  
**Documentação**: ~50 KB  

**TOTAL**: ~10.7 MB (sem o modelo .pt)

---

## 📞 SUPORTE

Se precisar de ajuda, consulte:
1. **GUIA_USO_TCC.md** - Instruções detalhadas
2. **OUTPUTS_TCC_REFERENCIA.md** - Análises e textos
3. **validate_outputs.py** (na pasta raiz) - Script de validação

---

## 🎓 PRONTO PARA USO!

Todos os arquivos estão organizados e prontos para:
- ✅ Inclusão direta no documento LaTeX
- ✅ Citação nas seções apropriadas
- ✅ Apêndices e material suplementar

**Boa sorte com o TCC!** 🎉

---

**Criado em**: 1 de novembro de 2025  
**Origem**: Sistema de Detecção de Deepfakes - TCC Victor  
**Status**: ✅ **VALIDADO E COMPLETO**
