# 🎓 GUIA RÁPIDO - USANDO OS OUTPUTS NO TCC

**Data**: 1 de novembro de 2025  
**Status**: ✅ **TODOS OS ARQUIVOS VALIDADOS E PRONTOS**

---

## ✅ CHECKLIST DE VALIDAÇÃO

### Arquivos Gerados
- [x] **4 Figuras** em `outputs/figures/` (2.6 MB total, DPI 300)
- [x] **2 Relatórios** em `outputs/reports/` (table_metrics.csv + run_report.md)
- [x] **2 Métricas brutas** (metrics_train.csv + metrics_cross.csv)
- [x] **4 Heatmaps** de exemplo em `outputs/heatmaps/`
- [x] **1 Modelo** treinado em `models/model_best.pt` (93.4 MB)

### Validação de Dados
- [x] **20 épocas** de treinamento completas
- [x] **Melhor Val AUC**: 85.07% (época 17) ✅
- [x] **2 datasets** validados (FaceForensics++ e Celeb-DF-v2)
- [x] **7.579 amostras** testadas no total
- [x] **Consistência** entre todos os arquivos ✅

---

## 📊 FIGURAS PARA O TCC

### 1. **training_curves.png** (445 KB)
**Onde usar**: Seção 4.1 - Resultados do Treinamento

**Legenda sugerida**:
> Figura X: Curvas de treinamento do modelo ao longo de 20 épocas. (a) Loss de treino e validação, (b) AUC de validação com linha de meta em 0.85, (c) F1-Score de validação, (d) Taxa de aprendizado com escala logarítmica. O melhor desempenho foi alcançado na época 17 (AUC: 85.07%, F1: 92.69%).

**Dados principais**:
- Val AUC: **85.07%** (época 17)
- Val F1: **92.69%** (época 17)
- Train Loss final: **0.0038** (overfitting moderado)

---

### 2. **f1_by_dataset.png** (106 KB)
**Onde usar**: Seção 4.2 - Cross-Dataset Evaluation

**Legenda sugerida**:
> Figura Y: Comparação de F1-Score e AUC entre os datasets FaceForensics++ e Celeb-DF-v2 no conjunto de teste. Observa-se F1-Score consistente (~92.9%) em ambos, mas AUC superior em FaceForensics++ (83.70% vs 73.09%), indicando possível overfitting ao estilo deste dataset.

**Dados principais**:
- FaceForensics++: AUC **83.70%**, F1 **92.87%** (1.050 amostras)
- Celeb-DF-v2: AUC **73.09%**, F1 **92.91%** (6.529 amostras)
- Gap: **10.6%** (esperado em cross-dataset)

---

### 3. **confusion_matrix.png** (133 KB)
**Onde usar**: Seção 4.3 - Análise de Erros

**Legenda sugerida**:
> Figura Z: Matrizes de confusão para os datasets (a) FaceForensics++ e (b) Celeb-DF-v2. Alto recall em ambos (95.56% e 98.81%) indica baixa taxa de falsos negativos, enquanto precision moderada (90.34% e 87.68%) sugere alguns falsos positivos, comportamento esperado em detecção de deepfakes onde prioriza-se evitar fakes não detectados.

**Análise**:
- **Recall altíssimo** (95-98%) → Poucos deepfakes passam despercebidos ✅
- **Precision boa** (87-90%) → Alguns vídeos reais marcados como fake
- **Trade-off aceitável** para aplicação de segurança

---

### 4. **gradcam_examples.png** (1.9 MB)
**Onde usar**: Seção 4.4 - Interpretabilidade Visual

**Legenda sugerida**:
> Figura W: Exemplos de mapas de atenção Grad-CAM para um vídeo deepfake do tipo NeuralTextures (FaceForensics++). Cada linha mostra: frame original, heatmap de atenção e sobreposição. O modelo foca predominantemente em regiões faciais (olhos, boca, bordas) sem depender excessivamente de artefatos de background, demonstrando aprendizado de padrões relevantes.

**Estatísticas**:
- Atenção média: **0.0463**
- Atenção máxima: **0.1896**
- Predição: **FAKE** (92.06% de confiança)
- Taxa de detecção facial: **100%**

---

## 📄 TABELAS PARA O TCC

### Tabela 1: Métricas de Treinamento

| Métrica | Valor | Época |
|---------|-------|-------|
| **Val AUC** | **85.07%** | 17 |
| **Val F1-Score** | **92.69%** | 17 |
| Val Loss | 0.5274 | 17 |
| Train Loss | 0.0038 | 20 |
| Learning Rate final | 0.0000125 | 20 |

**Fonte**: `outputs/metrics_train.csv`

---

### Tabela 2: Cross-Dataset Evaluation

| Dataset | AUC | F1-Score | Accuracy | Precision | Recall | Amostras |
|---------|-----|----------|----------|-----------|--------|----------|
| **FaceForensics++** | 83.70% | 92.87% | 87.43% | 90.34% | 95.56% | 1.050 |
| **Celeb-DF-v2** | 73.09% | 92.91% | 86.98% | 87.68% | 98.81% | 6.529 |
| **Média Ponderada** | **74.56%** | **92.91%** | **87.02%** | **88.16%** | **98.15%** | **7.579** |

**Fonte**: `outputs/metrics_cross.csv`

---

## 📝 TEXTOS PRONTOS PARA O TCC

### Seção: Resultados do Treinamento

> O modelo foi treinado por 20 épocas utilizando os datasets FaceForensics++ e Celeb-DF-v2, totalizando 13.529 vídeos distribuídos em 70% treino, 15% validação e 15% teste. O melhor desempenho foi alcançado na época 17, com **AUC de validação de 85.07%** e **F1-Score de 92.69%**, superando a meta estabelecida de 80% para o AUC (Figura X). 
>
> A Figura X apresenta as curvas de evolução das métricas ao longo do treinamento. Observa-se convergência adequada com Train Loss final de 0.0038, embora haja sinais de overfitting moderado evidenciados pela estabilização do Val Loss em 0.5274 enquanto o Train Loss continua decrescendo. O scheduler ReduceLROnPlateau reduziu a taxa de aprendizado de 1e-4 para 1.25e-5 ao longo do treinamento, contribuindo para a estabilização do modelo.

### Seção: Cross-Dataset Evaluation

> Para avaliar a capacidade de generalização do modelo, realizou-se uma avaliação cross-dataset nos conjuntos de teste de FaceForensics++ (1.050 amostras) e Celeb-DF-v2 (6.529 amostras). Os resultados estão apresentados na Tabela 2 e Figura Y.
>
> O modelo alcançou **AUC de 83.70% em FaceForensics++** e **73.09% em Celeb-DF-v2**, com F1-Score consistente de aproximadamente 92.9% em ambos os datasets. O gap de 10.6% entre os AUCs é esperado em avaliações cross-dataset e indica possível overfitting ao estilo de deepfakes do FaceForensics++, que representa 51.7% do dataset de treinamento.
>
> Destaca-se o **recall médio de 98.15%**, indicando que o modelo raramente deixa passar deepfakes (baixa taxa de falsos negativos), comportamento desejável para aplicações de segurança. A precision de 88.16% sugere alguns falsos positivos, mas esse trade-off é aceitável dado o contexto de aplicação.

### Seção: Análise de Erros

> As matrizes de confusão (Figura Z) revelam padrões interessantes nos erros do modelo. Em FaceForensics++, o recall de 95.56% indica que apenas 4.44% dos deepfakes não foram detectados, enquanto em Celeb-DF-v2 esse valor cai para 1.19% (recall de 98.81%).
>
> Os falsos positivos (vídeos reais classificados como fake) representam aproximadamente 10-13% das predições de fake, o que pode estar relacionado a vídeos reais de baixa qualidade ou com artefatos de compressão similares aos gerados por técnicas de síntese. Esta análise sugere que ajustes no threshold de decisão ou técnicas de calibração de probabilidades poderiam melhorar a precision sem sacrificar significativamente o recall.

### Seção: Interpretabilidade Visual

> A interpretabilidade do modelo foi avaliada através da técnica Grad-CAM (Gradient-weighted Class Activation Mapping), que gera mapas de atenção visual destacando regiões importantes para a decisão. A Figura W apresenta exemplos de heatmaps gerados para um vídeo deepfake do tipo NeuralTextures do dataset FaceForensics++.
>
> Os resultados mostram que o modelo foca predominantemente em **regiões faciais** (olhos, boca, bordas da face), com atenção média de 0.0463 e máxima de 0.1896 em áreas específicas. Observa-se que o modelo **não depende excessivamente de artefatos de background** ou elementos não-faciais, demonstrando que a rede aprendeu padrões relevantes relacionados às características do rosto manipulado.
>
> A variação temporal da atenção entre frames consecutivos (desvio padrão de 0.0561) sugere que o componente LSTM está capturando inconsistências temporais, uma característica importante para detecção de deepfakes que não seria capturada por abordagens baseadas apenas em frames individuais.

---

## 🔬 METODOLOGIA - ESPECIFICAÇÕES TÉCNICAS

### Hardware e Software

> Os experimentos foram conduzidos em uma estação de trabalho equipada com GPU NVIDIA GeForce RTX 4060 (8GB VRAM), processador Intel Core i7 e 32GB de RAM. O sistema operacional utilizado foi Windows 11, com CUDA 12.1 para aceleração por GPU.
>
> O modelo foi implementado em Python 3.11.9 utilizando PyTorch 2.5.1 como framework de deep learning. Para detecção facial, utilizou-se o detector MTCNN (Multi-task Cascaded Convolutional Networks) da biblioteca facenet-pytorch 2.6.0.

### Arquitetura do Modelo

> A arquitetura proposta combina uma rede neural convolucional (CNN) para extração de features espaciais com uma rede LSTM bidirecional para modelagem temporal. A CNN baseia-se na ResNet-34 pré-treinada no ImageNet, adaptada para processar sequências de 16 frames por vídeo.
>
> As features extraídas pela CNN (512 dimensões por frame) são alimentadas em uma LSTM bidirecional com 512 unidades ocultas e 2 camadas, resultando em 1024 features após concatenação das direções forward e backward. Uma camada totalmente conectada final produz a probabilidade de o vídeo ser um deepfake. O modelo possui **24.4 milhões de parâmetros** no total.

### Configuração de Treinamento

> O treinamento foi realizado com batch size de 8 vídeos por mini-batch, otimizador Adam com taxa de aprendizado inicial de 1e-4, e função de perda Binary Cross-Entropy with Logits (BCEWithLogitsLoss) com pos_weight de 0.167 para balancear as classes (proporção real/fake de 1:6 no dataset).
>
> Utilizou-se o scheduler ReduceLROnPlateau com paciência de 3 épocas para reduzir a taxa de aprendizado em 50% quando o AUC de validação estagnava. Early stopping com paciência de 5 épocas foi aplicado, mas o treinamento completou todas as 20 épocas planejadas. Mixed precision training (FP16) foi habilitado para otimizar o uso de memória GPU.
>
> O tempo total de treinamento foi de **38 horas e 45 minutos** (aproximadamente 2 horas por época).

---

## 📚 COMO INSERIR NO LATEX

### Inserir Figura

```latex
\begin{figure}[htb]
    \centering
    \includegraphics[width=0.95\textwidth]{outputs/figures/training_curves.png}
    \caption{Curvas de treinamento do modelo ao longo de 20 épocas. 
    (a) Loss de treino e validação, (b) AUC de validação, 
    (c) F1-Score de validação, (d) Taxa de aprendizado.}
    \label{fig:training_curves}
\end{figure}
```

### Inserir Tabela (usando CSVautotabular)

```latex
\begin{table}[htb]
    \centering
    \caption{Métricas de cross-dataset evaluation.}
    \label{tab:cross_dataset}
    \csvautotabular{outputs/reports/table_metrics.csv}
\end{table}
```

### Inserir Tabela (manualmente)

```latex
\begin{table}[htb]
    \centering
    \caption{Resultados da avaliação cross-dataset.}
    \label{tab:cross_dataset}
    \begin{tabular}{lcccccc}
        \toprule
        \textbf{Dataset} & \textbf{AUC} & \textbf{F1} & \textbf{Acc} & \textbf{Prec} & \textbf{Rec} & \textbf{Amostras} \\
        \midrule
        FaceForensics++ & 0.8370 & 0.9287 & 0.8743 & 0.9034 & 0.9556 & 1.050 \\
        Celeb-DF-v2 & 0.7309 & 0.9291 & 0.8698 & 0.8768 & 0.9881 & 6.529 \\
        \midrule
        \textbf{Média Ponderada} & \textbf{0.7456} & \textbf{0.9291} & \textbf{0.8702} & \textbf{0.8816} & \textbf{0.9815} & \textbf{7.579} \\
        \bottomrule
    \end{tabular}
\end{table}
```

### Referenciar no texto

```latex
Como pode ser observado na Figura~\ref{fig:training_curves}, o modelo 
alcançou convergência adequada na época 17 com AUC de validação de 85.07\%.

A Tabela~\ref{tab:cross_dataset} apresenta os resultados da avaliação 
cross-dataset, evidenciando F1-Score consistente de aproximadamente 92.9\% 
em ambos os datasets testados.
```

---

## ✅ CHECKLIST FINAL PARA O TCC

### Antes de Submeter

- [ ] Copiar todas as 4 figuras de `outputs/figures/` para pasta de imagens do LaTeX
- [ ] Verificar resolução das figuras (devem estar em 300 DPI)
- [ ] Inserir legendas completas e descritivas em cada figura
- [ ] Adicionar referências no texto para todas as figuras e tabelas
- [ ] Verificar consistência dos valores citados no texto com as tabelas
- [ ] Incluir `table_metrics.csv` como tabela no apêndice (opcional)
- [ ] Citar o relatório técnico (`run_report.md`) como documentação adicional
- [ ] Validar que todas as métricas citadas estão corretas

### Arquivos a Incluir

**Obrigatórios** (no documento):
- [x] 4 figuras PNG em alta resolução
- [x] 2-3 tabelas com métricas principais
- [x] Textos descritivos adaptados

**Opcionais** (apêndice ou material suplementar):
- [ ] `table_metrics.csv` - Tabela completa de métricas
- [ ] `metrics_train.csv` - Histórico completo de treino
- [ ] `metrics_cross.csv` - Detalhes cross-dataset
- [ ] `run_report.md` - Relatório técnico completo
- [ ] Exemplos de heatmaps individuais do Grad-CAM

---

## 📞 VALIDAÇÃO FINAL

**Para garantir que tudo está correto, execute**:

```bash
python validate_outputs.py
```

**Saída esperada**: ✅ VALIDAÇÃO COMPLETA: TODOS OS OUTPUTS ESTÃO OK!

---

## 🎉 PRONTO!

Todos os outputs foram:
- ✅ **Limpos** (removidos dados antigos/inconsistentes)
- ✅ **Regenerados** (a partir dos dados brutos validados)
- ✅ **Validados** (consistência verificada)
- ✅ **Documentados** (guias e textos prontos)

**Agora você pode focar em**:
1. Adaptar os textos ao seu estilo de escrita
2. Inserir as figuras e tabelas no LaTeX
3. Revisar as referências e citações
4. Finalizar o documento do TCC

**Boa sorte com o TCC! 🎓**

---

**Criado em**: 1 de novembro de 2025  
**Validado**: ✅ Todos os 11 arquivos obrigatórios presentes e corretos
