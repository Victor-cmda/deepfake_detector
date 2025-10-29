# 🛠️ Scripts Auxiliares - Deepfake Detector

Scripts utilitários para preparação de dados e visualizações.

## 📊 Scripts de Preparação de Dados

### `organize_datasets.py`
Organiza os datasets baixados na estrutura correta.

**Uso**:
```bash
python scripts/organize_datasets.py
```

**Funcionalidade**:
- Cria estrutura de pastas para cada dataset
- Organiza vídeos em `videos_real/` e `videos_fake/`
- Gera arquivos de índice CSV

---

### `generate_splits.py`
Gera divisões treino/validação/teste para os datasets.

**Uso**:
```bash
python scripts/generate_splits.py
```

**Output**: `data/splits_faceforensicspp.csv`

---

### `validate_splits.py`
Valida que as divisões foram criadas corretamente.

**Uso**:
```bash
python scripts/validate_splits.py
```

---

## 🎨 Scripts de Visualização

### `create_sample_videos.py`
Cria vídeos de exemplo sintéticos para testes.

**Uso**:
```bash
python scripts/create_sample_videos.py
```

**Output**: Vídeos em `data/{dataset}/videos_{real|fake}/`

---

### `create_preprocessing_viz.py`
Gera visualizações do pipeline de pré-processamento.

**Uso**:
```bash
python scripts/create_preprocessing_viz.py
```

**Output**: Figuras em `outputs/figures/`

---

### `create_model_diagram.py`
Gera diagrama da arquitetura do modelo.

**Uso**:
```bash
python scripts/create_model_diagram.py
```

**Output**: Diagrama da arquitetura CNN-LSTM

---

## 🔄 Ordem de Execução Recomendada

Para configurar o projeto do zero:

1. **Organizar datasets**:
   ```bash
   python scripts/organize_datasets.py
   ```

2. **Gerar divisões**:
   ```bash
   python scripts/generate_splits.py
   ```

3. **Validar divisões**:
   ```bash
   python scripts/validate_splits.py
   ```

4. **Criar visualizações** (opcional):
   ```bash
   python scripts/create_preprocessing_viz.py
   python scripts/create_model_diagram.py
   ```

---

## 📝 Notas

- Estes scripts devem ser executados **uma vez** durante a configuração inicial
- Alguns scripts requerem que os datasets estejam baixados em `data/`
- Scripts de visualização geram arquivos em `outputs/`

---

**Data de Organização**: 29 de outubro de 2025
