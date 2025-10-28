# Instruções para Organização de Datasets

## Estrutura de Diretórios

Este diretório contém três datasets principais para detecção de deepfakes:

```
data/
├── FaceForensics++/
│   ├── videos_real/    <- Colocar vídeos REAIS aqui
│   └── videos_fake/    <- Colocar vídeos FAKE aqui
├── Celeb-DF-v2/
│   ├── videos_real/
│   └── videos_fake/
├── WildDeepfake/
│   ├── videos_real/
│   └── videos_fake/
├── index_faceforensicspp.csv    (gerado automaticamente)
├── index_celebdf.csv            (gerado automaticamente)
└── index_wilddeepfake.csv       (gerado automaticamente)
```

## Instruções de Uso

### 1. Baixar os Datasets

**FaceForensics++:**
- Site: https://github.com/ondyari/FaceForensics
- Requisitar acesso e baixar vídeos

**Celeb-DF-v2:**
- Site: https://github.com/yuezunli/celeb-deepfakeforensics
- Baixar versão 2

**WildDeepfake:**
- Site: https://github.com/deepfakeinthewild/deepfake-in-the-wild
- Baixar dataset

### 2. Organizar os Vídeos

Copie os vídeos para as pastas apropriadas:
- Vídeos reais/originais → `videos_real/`
- Vídeos deepfake/manipulados → `videos_fake/`

Formatos suportados: `.mp4`, `.avi`, `.mov`, `.mkv`

### 3. Indexar os Datasets

Execute o script de organização na raiz do projeto:

```bash
python organize_datasets.py
```

Este script irá:
- Verificar a estrutura de diretórios
- Contar os vídeos em cada categoria
- Gerar arquivos CSV de índice com informações:
  - Caminho do vídeo
  - Label (0=real, 1=fake)
  - Nome do dataset
  - Número de frames

### 4. Arquivos Gerados

Os seguintes arquivos CSV serão criados/atualizados:
- `index_faceforensicspp.csv`
- `index_celebdf.csv`
- `index_wilddeepfake.csv`

Estes arquivos são utilizados pelas etapas seguintes do projeto (divisão treino/teste, treinamento, etc.)

## Observações

- Certifique-se de ter espaço em disco suficiente (datasets podem ocupar centenas de GB)
- O script pode ser executado múltiplas vezes - ele sempre sobrescreve os arquivos CSV
- Se não houver vídeos, arquivos CSV vazios serão criados com as colunas corretas
