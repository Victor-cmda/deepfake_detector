"""
Visualização dos resultados do treinamento completo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Carregar métricas
df = pd.read_csv('outputs/metrics_train.csv')

# Criar figura com 6 subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('📊 Resultados do Treinamento Completo - Deepfake Detector', fontsize=16, fontweight='bold')

# 1. Train Loss vs Val Loss
ax1 = axes[0, 0]
ax1.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss', color='#2E86AB', linewidth=2, markersize=6)
ax1.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss', color='#A23B72', linewidth=2, markersize=6)
ax1.axvline(x=17, color='green', linestyle='--', alpha=0.5, label='Melhor Época')
ax1.set_xlabel('Época')
ax1.set_ylabel('Loss')
ax1.set_title('Loss (Train vs Validation)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Val AUC
ax2 = axes[0, 1]
ax2.plot(df['epoch'], df['val_auc'] * 100, 'o-', color='#F18F01', linewidth=2, markersize=6)
ax2.axvline(x=17, color='green', linestyle='--', alpha=0.5, label='Melhor Época (85.07%)')
ax2.axhline(y=85.07, color='green', linestyle=':', alpha=0.5)
ax2.axhline(y=80, color='red', linestyle='--', alpha=0.3, label='Meta (80%)')
ax2.set_xlabel('Época')
ax2.set_ylabel('AUC (%)')
ax2.set_title('Validation AUC')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(60, 90)

# 3. Val F1-Score
ax3 = axes[0, 2]
ax3.plot(df['epoch'], df['val_f1'] * 100, 'o-', color='#C73E1D', linewidth=2, markersize=6)
ax3.axvline(x=17, color='green', linestyle='--', alpha=0.5, label='Melhor Época (92.69%)')
ax3.axhline(y=92.69, color='green', linestyle=':', alpha=0.5)
ax3.axhline(y=85, color='red', linestyle='--', alpha=0.3, label='Meta (85%)')
ax3.set_xlabel('Época')
ax3.set_ylabel('F1-Score (%)')
ax3.set_title('Validation F1-Score')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(50, 100)

# 4. Learning Rate Schedule
ax4 = axes[1, 0]
ax4.plot(df['epoch'], df['learning_rate'], 'o-', color='#6A0572', linewidth=2, markersize=6)
ax4.set_xlabel('Época')
ax4.set_ylabel('Learning Rate')
ax4.set_title('Learning Rate Schedule')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

# 5. Train Loss (zoom)
ax5 = axes[1, 1]
ax5.plot(df['epoch'], df['train_loss'], 'o-', color='#2E86AB', linewidth=2, markersize=6)
ax5.axvline(x=17, color='green', linestyle='--', alpha=0.5, label='Melhor Época')
ax5.set_xlabel('Época')
ax5.set_ylabel('Train Loss')
ax5.set_title('Train Loss (Convergência)')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_yscale('log')

# 6. Comparação de Métricas (Época 17)
ax6 = axes[1, 2]
epoch_17 = df[df['epoch'] == 17].iloc[0]
metrics = {
    'Val AUC': epoch_17['val_auc'] * 100,
    'Val F1': epoch_17['val_f1'] * 100,
    'Train Loss': epoch_17['train_loss'] * 100,
    'Val Loss': epoch_17['val_loss'] * 100
}
colors = ['#F18F01', '#C73E1D', '#2E86AB', '#A23B72']
bars = ax6.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.set_ylabel('Valor (%)')
ax6.set_title('Métricas da Melhor Época (17)')
ax6.grid(True, axis='y', alpha=0.3)
# Adicionar valores nas barras
for bar in bars:
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/training_results.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo: outputs/figures/training_results.png")

# Criar segunda figura: Análise detalhada
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('📈 Análise Detalhada do Treinamento', fontsize=14, fontweight='bold')

# Gap Train/Val Loss
ax1 = axes2[0]
gap = df['val_loss'] - df['train_loss']
ax1.plot(df['epoch'], gap, 'o-', color='#A23B72', linewidth=2, markersize=6)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax1.axvline(x=17, color='green', linestyle='--', alpha=0.5, label='Melhor Época')
ax1.set_xlabel('Época')
ax1.set_ylabel('Val Loss - Train Loss')
ax1.set_title('Gap de Generalização (Overfitting)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Estabilidade do Val AUC
ax2 = axes2[1]
rolling_mean = df['val_auc'].rolling(window=3).mean()
rolling_std = df['val_auc'].rolling(window=3).std()
ax2.plot(df['epoch'], df['val_auc'] * 100, 'o-', alpha=0.5, label='Val AUC', color='#F18F01')
ax2.plot(df['epoch'], rolling_mean * 100, '-', linewidth=2, label='Média Móvel (3 épocas)', color='#F18F01')
ax2.fill_between(df['epoch'], 
                  (rolling_mean - rolling_std) * 100, 
                  (rolling_mean + rolling_std) * 100, 
                  alpha=0.2, color='#F18F01')
ax2.axvline(x=17, color='green', linestyle='--', alpha=0.5, label='Melhor Época')
ax2.set_xlabel('Época')
ax2.set_ylabel('AUC (%)')
ax2.set_title('Estabilidade do Val AUC')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/training_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo: outputs/figures/training_analysis.png")

# Imprimir resumo
print("\n" + "=" * 60)
print("📊 RESUMO DO TREINAMENTO")
print("=" * 60)
print(f"\n🏆 Melhor Época: 17")
print(f"   - Val AUC: {epoch_17['val_auc']*100:.2f}%")
print(f"   - Val F1: {epoch_17['val_f1']*100:.2f}%")
print(f"   - Val Loss: {epoch_17['val_loss']:.4f}")
print(f"   - Train Loss: {epoch_17['train_loss']:.4f}")
print(f"   - Learning Rate: {epoch_17['learning_rate']:.2e}")

print(f"\n📈 Evolução:")
print(f"   - AUC: {df.iloc[0]['val_auc']*100:.2f}% → {epoch_17['val_auc']*100:.2f}% (+{(epoch_17['val_auc'] - df.iloc[0]['val_auc'])*100:.2f}%)")
print(f"   - F1: {df.iloc[0]['val_f1']*100:.2f}% → {epoch_17['val_f1']*100:.2f}% (+{(epoch_17['val_f1'] - df.iloc[0]['val_f1'])*100:.2f}%)")
print(f"   - Train Loss: {df.iloc[0]['train_loss']:.4f} → {df.iloc[-1]['train_loss']:.4f}")

print(f"\n⚠️  Overfitting:")
print(f"   - Gap (Época 17): {epoch_17['val_loss'] - epoch_17['train_loss']:.4f}")
print(f"   - Gap (Época 1): {df.iloc[0]['val_loss'] - df.iloc[0]['train_loss']:.4f}")
print(f"   - Gap (Final): {df.iloc[-1]['val_loss'] - df.iloc[-1]['train_loss']:.4f}")

print("\n" + "=" * 60)
print("✅ Visualizações geradas com sucesso!")
print("=" * 60)
