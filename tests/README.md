# 🧪 Scripts de Teste - Deepfake Detector

Esta pasta contém todos os scripts de teste e validação do projeto.

## 📋 Índice de Scripts

### ✅ Testes de Ambiente

- **`test_environment.py`**: Validação completa do ambiente (imports, GPU, OpenCV, MTCNN, modelo)
- **`check_cuda.py`**: Verificação rápida de disponibilidade de GPU/CUDA

### 🧠 Testes de Modelo

- **`test_model.py`**: Testa criação e forward pass do modelo
- **`test_quick_run.py`**: Treinamento rápido (2 épocas) para validação
- **`test_task_8.py`**: Validação específica da Tarefa 8

### 📊 Testes de Dados

- **`test_dataloader.py`**: Validação do pipeline de carregamento de dados
- **`test_preprocessing.py`**: Testa preprocessamento de vídeos e detecção facial

### 🖥️ Testes de Interface

- **`test_interface_fix.py`**: Teste das correções da interface Gradio
- **`test_cudnn_fix.py`**: Validação da correção do erro CuDNN RNN

---

## 🚀 Como Usar

### Teste Completo do Ambiente
```bash
python tests/test_environment.py
```

### Verificação Rápida de GPU
```bash
python tests/check_cuda.py
```

### Teste de Treinamento Rápido
```bash
python tests/test_quick_run.py
```

### Teste da Interface (após correções)
```bash
python tests/test_cudnn_fix.py
```

---

## 📦 Requisitos

Todos os testes assumem que:
- ✅ Ambiente virtual ativo (`.venv-1`)
- ✅ Dependências instaladas (`requirements.txt`)
- ✅ Datasets organizados (`data/`)
- ✅ Modelo treinado (`models/model_best.pt`) para alguns testes

---

## 🎯 Propósito

Estes scripts são **auxiliares de desenvolvimento** e **não fazem parte do pipeline principal** do projeto. Eles servem para:

1. **Validar** que o ambiente está configurado corretamente
2. **Testar** componentes individuais durante desenvolvimento
3. **Depurar** problemas específicos
4. **Verificar** que correções funcionam como esperado

---

## 📝 Notas

- Scripts de teste **não devem** ser executados em produção
- Alguns testes requerem que o modelo já esteja treinado
- Testes podem gerar outputs temporários (logs, figuras, etc.)
- Use testes individuais para isolar problemas

---

**Data de Organização**: 29 de outubro de 2025
