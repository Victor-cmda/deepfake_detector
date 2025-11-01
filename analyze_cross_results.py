"""
Visualização dos resultados da avaliação cross-dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

# Carregar métricas
df = pd.read_csv('outputs/metrics_cross.csv')

# Filtrar apenas FaceForensics++ e Celeb-DF (WildDeepfake tem 0)
df_valid = df[df['dataset'] != 'WildDeepfake'].copy()

print("=" * 70)
print("📊 RESULTADOS DA AVALIAÇÃO CROSS-DATASET")
print("=" * 70)
print()

# Imprimir tabela
print(df_valid.to_string(index=False))
print()

# Criar visualização
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('📊 Avaliação Cross-Dataset - Deepfake Detector', fontsize=16, fontweight='bold', y=0.995)

# Cores para cada dataset
colors = {'FaceForensics++': '#2E86AB', 'Celeb-DF-v2': '#A23B72'}

# 1. Comparação de Métricas
ax1 = axes[0, 0]
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
x = range(len(metrics_to_plot))
width = 0.35

for i, (dataset, color) in enumerate(colors.items()):
    row = df_valid[df_valid['dataset'] == dataset].iloc[0]
    values = [row[m] * 100 for m in metrics_to_plot]
    ax1.bar([xi + i * width for xi in x], values, width, 
            label=dataset, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_ylabel('Score (%)')
ax1.set_title('Comparação de Métricas por Dataset')
ax1.set_xticks([xi + width/2 for xi in x])
ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)
ax1.set_ylim(0, 100)

# Adicionar valores nas barras
for i, (dataset, color) in enumerate(colors.items()):
    row = df_valid[df_valid['dataset'] == dataset].iloc[0]
    values = [row[m] * 100 for m in metrics_to_plot]
    for j, v in enumerate(values):
        ax1.text(j + i * width, v + 1, f'{v:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. AUC Comparison
ax2 = axes[0, 1]
datasets = df_valid['dataset'].tolist()
aucs = (df_valid['auc'] * 100).tolist()
bars = ax2.barh(datasets, aucs, color=[colors[d] for d in datasets], 
                alpha=0.8, edgecolor='black', linewidth=2)
ax2.axvline(x=80, color='red', linestyle='--', alpha=0.5, label='Meta (80%)')
ax2.set_xlabel('AUC (%)')
ax2.set_title('AUC por Dataset')
ax2.legend()
ax2.grid(True, axis='x', alpha=0.3)
ax2.set_xlim(0, 100)

# Adicionar valores
for i, (bar, auc) in enumerate(zip(bars, aucs)):
    ax2.text(auc + 1, i, f'{auc:.2f}%', va='center', fontweight='bold')

# 3. F1-Score Comparison
ax3 = axes[0, 2]
datasets = df_valid['dataset'].tolist()
f1s = (df_valid['f1'] * 100).tolist()
bars = ax3.barh(datasets, f1s, color=[colors[d] for d in datasets], 
                alpha=0.8, edgecolor='black', linewidth=2)
ax3.axvline(x=85, color='red', linestyle='--', alpha=0.5, label='Meta (85%)')
ax3.set_xlabel('F1-Score (%)')
ax3.set_title('F1-Score por Dataset')
ax3.legend()
ax3.grid(True, axis='x', alpha=0.3)
ax3.set_xlim(0, 100)

# Adicionar valores
for i, (bar, f1) in enumerate(zip(bars, f1s)):
    ax3.text(f1 + 1, i, f'{f1:.2f}%', va='center', fontweight='bold')

# 4. Precision vs Recall
ax4 = axes[1, 0]
for dataset, color in colors.items():
    row = df_valid[df_valid['dataset'] == dataset].iloc[0]
    ax4.scatter(row['recall'] * 100, row['precision'] * 100, 
               s=300, color=color, alpha=0.7, edgecolor='black', linewidth=2,
               label=dataset)
    ax4.annotate(dataset, 
                (row['recall'] * 100, row['precision'] * 100),
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontweight='bold')

ax4.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Linha de Referência')
ax4.set_xlabel('Recall (%)')
ax4.set_ylabel('Precision (%)')
ax4.set_title('Precision vs Recall')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(85, 100)
ax4.set_ylim(85, 100)

# 5. Número de Amostras
ax5 = axes[1, 1]
datasets = df_valid['dataset'].tolist()
samples = df_valid['total_samples'].tolist()
bars = ax5.bar(datasets, samples, color=[colors[d] for d in datasets], 
              alpha=0.8, edgecolor='black', linewidth=2)
ax5.set_ylabel('Número de Amostras')
ax5.set_title('Tamanho dos Datasets de Teste')
ax5.grid(True, axis='y', alpha=0.3)

# Adicionar valores
for bar, sample in zip(bars, samples):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{sample:,}',
            ha='center', va='bottom', fontweight='bold')

# 6. Radar Chart de Métricas
ax6 = axes[1, 2]
ax6.axis('off')

# Criar tabela resumida
table_data = []
for _, row in df_valid.iterrows():
    table_data.append([
        row['dataset'],
        f"{row['accuracy']*100:.2f}%",
        f"{row['precision']*100:.2f}%",
        f"{row['recall']*100:.2f}%",
        f"{row['f1']*100:.2f}%",
        f"{row['auc']*100:.2f}%"
    ])

table = ax6.table(cellText=table_data,
                 colLabels=['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
                 cellLoc='center',
                 loc='center',
                 colColours=['lightgray']*6)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Colorir linhas
for i, dataset in enumerate(df_valid['dataset']):
    for j in range(6):
        cell = table[(i+1, j)]
        cell.set_facecolor(colors[dataset] if j == 0 else 'white')
        cell.set_alpha(0.3)

ax6.set_title('Tabela Resumida de Métricas', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/figures/cross_dataset_summary.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico salvo: outputs/figures/cross_dataset_summary.png")

# Imprimir análise
print("\n" + "=" * 70)
print("📈 ANÁLISE DOS RESULTADOS")
print("=" * 70)

for _, row in df_valid.iterrows():
    dataset = row['dataset']
    print(f"\n🎯 {dataset}:")
    print(f"   Accuracy:  {row['accuracy']*100:.2f}%")
    print(f"   Precision: {row['precision']*100:.2f}%")
    print(f"   Recall:    {row['recall']*100:.2f}%")
    print(f"   F1-Score:  {row['f1']*100:.2f}%")
    print(f"   AUC:       {row['auc']*100:.2f}%")
    print(f"   Amostras:  {row['total_samples']:,}")
    
    # Análise
    if row['auc'] >= 0.80:
        print(f"   ✅ Excelente generalização (AUC ≥ 80%)")
    elif row['auc'] >= 0.70:
        print(f"   ✅ Boa generalização (AUC ≥ 70%)")
    else:
        print(f"   ⚠️  Generalização moderada")

print("\n" + "=" * 70)
print("🏆 DESEMPENHO GERAL")
print("=" * 70)

# Calcular média ponderada
total_samples = df_valid['total_samples'].sum()
weighted_auc = (df_valid['auc'] * df_valid['total_samples']).sum() / total_samples
weighted_f1 = (df_valid['f1'] * df_valid['total_samples']).sum() / total_samples

print(f"\nMédia Ponderada (por número de amostras):")
print(f"  - AUC:      {weighted_auc*100:.2f}%")
print(f"  - F1-Score: {weighted_f1*100:.2f}%")
print(f"  - Total de amostras testadas: {total_samples:,}")

# Comparação FaceForensics++ vs Celeb-DF
ff_auc = df_valid[df_valid['dataset'] == 'FaceForensics++']['auc'].values[0]
cd_auc = df_valid[df_valid['dataset'] == 'Celeb-DF-v2']['auc'].values[0]
diff_auc = (ff_auc - cd_auc) * 100

print(f"\n📊 Diferença de Generalização:")
print(f"  - FaceForensics++ AUC: {ff_auc*100:.2f}%")
print(f"  - Celeb-DF AUC:        {cd_auc*100:.2f}%")
print(f"  - Diferença:           {diff_auc:.2f}% (FF++ melhor)")

if diff_auc < 5:
    print(f"  ✅ Excelente consistência entre datasets!")
elif diff_auc < 10:
    print(f"  ✅ Boa consistência entre datasets")
else:
    print(f"  ⚠️  Diferença significativa - possível overfitting ao FF++")

print("\n" + "=" * 70)
