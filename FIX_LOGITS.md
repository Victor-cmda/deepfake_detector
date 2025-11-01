# CORREÇÃO CRÍTICA - BCEWithLogitsLoss

## 🚨 Problema Identificado

**Treinamento anterior falhou completamente:**
- Train Loss: 1.80 (deveria ser ~0.30-0.40)
- Val AUC: 0.53 (random guess = 0.50)
- Val F1: 0.9231 (travado - sempre prevendo FAKE)
- Modelo não aprendeu nada!

### Causa Raiz:
```python
# ❌ ERRADO:
model.forward() → retorna sigmoid(logits)  # probabilidades [0,1]
criterion = BCEWithLogitsLoss()            # espera logits [-∞,+∞]

# Resultado: loss explode porque sigmoid(probabilidades) não são logits!
```

---

## ✅ Correções Implementadas

### 1. Modelo Retorna Logits Agora

**Arquivo:** `src/model.py`

```python
class DeepfakeDetector(nn.Module):
    def __init__(self, ..., return_logits=True):
        self.return_logits = return_logits
    
    def forward(self, x):
        # ...
        logits = self.fc(x)  # (batch_size, 1) - raw scores
        
        if self.return_logits:
            return logits  # ✅ Para BCEWithLogitsLoss
        else:
            return self.sigmoid(logits)  # Para inference
```

**Mudança:**
- ✅ `return_logits=True` (padrão): retorna logits para treinamento
- ✅ `return_logits=False`: retorna probabilidades para inference/avaliação

### 2. pos_weight Corrigido

**Arquivo:** `src/train.py`

```python
# ❌ ANTES (errado):
pos_weight = weight_real / weight_fake  # = 6.0 (muito alto!)

# ✅ AGORA (correto):
pos_weight = num_real / num_fake  # = 700/4200 = 0.167
```

**Por quê?**
- `pos_weight` em `BCEWithLogitsLoss` penaliza positivos (FAKE=1)
- Como queremos balancear REAL (minoria), usamos pos_weight < 1
- Isso faz o modelo dar mais importância para acertar REAL

### 3. create_model Atualizado

**Arquivo:** `src/train.py`

```python
# ✅ NOVO:
model = create_model(
    num_frames=num_frames, 
    pretrained=True, 
    device=device, 
    return_logits=True  # ← Adicionado
)
```

---

## 📊 Resultados Esperados Agora

### Loss:
```
Época | Train Loss | Val Loss | Esperado
------|------------|----------|----------
  1   | ~0.60-0.70 | ~0.55-0.65| ✅ Decrescendo
  5   | ~0.35-0.45 | ~0.38-0.48| ✅ Convergindo
 10   | ~0.25-0.35 | ~0.30-0.40| ✅ Estável
```

### Métricas:
```
Val AUC: 0.53 → 0.70-0.85  ✅ Melhorando
Val F1:  0.92 → 0.75-0.85  ✅ Balanceado (ambas classes)
```

### Comportamento:
- ✅ Train loss diminui consistentemente
- ✅ Val loss segue train loss
- ✅ AUC aumenta progressivamente
- ✅ F1 não fica travado (prevê ambas classes)

---

## 🔧 O Que Foi Mudado

| Arquivo | Mudança | Linha(s) |
|---------|---------|----------|
| `src/model.py` | Adicionar `return_logits` param | 18, 21 |
| `src/model.py` | Modificar `forward()` para retornar logits | 107-113 |
| `src/model.py` | Adicionar `return_logits` em `create_model()` | 166, 182 |
| `src/train.py` | Passar `return_logits=True` em `create_model()` | 220 |
| `src/train.py` | Corrigir cálculo `pos_weight` | 233-241 |

---

## 🎯 Teste Rápido (5 min)

Para verificar se funcionou, rode **1 época apenas**:

```python
# No train_full.py, temporariamente:
config = {
    'num_epochs': 1,  # ← Teste
    # ...
}
```

**Checklist de Sucesso:**
- [ ] Train Loss < 0.70 na primeira época
- [ ] Val Loss < 0.65 na primeira época  
- [ ] Val AUC > 0.55 (melhor que random)
- [ ] Loss diminui ao longo dos batches

Se tudo OK, rodar treinamento completo (10-20 épocas).

---

## 🚀 Comando para Treinar

```cmd
.venv-1\Scripts\python.exe train_full.py
```

**Tempo estimado:** 10-15 horas (8 épocas × 1-2h)

---

## 📝 Compatibilidade com Código Antigo

**Interface/Inference:** Precisa usar `return_logits=False`

```python
# Para inference:
model = create_model(..., return_logits=False)
# OU
model.return_logits = False

# Agora model(x) retorna probabilidades [0,1]
```

**Modelos Salvos:** Modelos antigos funcionam! O parâmetro `return_logits` é adicionado com padrão `True`.

---

**Status:** ✅ CORREÇÕES CRÍTICAS APLICADAS  
**Data:** 30 de outubro de 2025  
**Pronto para:** Novo treinamento
