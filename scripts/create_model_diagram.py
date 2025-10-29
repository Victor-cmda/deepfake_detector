"""
Script para criar visualização da arquitetura do modelo.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.utils import ensure_dir


def create_architecture_diagram():
    """
    Cria diagrama visual da arquitetura DeepfakeDetector.
    """
    print("Criando diagrama da arquitetura...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Configurar eixos
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Título
    ax.text(5, 11.5, 'Arquitetura DeepfakeDetector', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(5, 11, 'ResNet-34 + BiLSTM + Sigmoid', 
            ha='center', va='center', fontsize=12, style='italic')
    
    # 1. INPUT
    rect = patches.FancyBboxPatch((0.5, 9.5), 2, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.5, 10.2, 'INPUT', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 9.9, '(B, 16, 3, 224, 224)', ha='center', va='center', fontsize=8)
    
    # Seta 1
    ax.arrow(1.5, 9.5, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # 2. RESHAPE
    rect = patches.FancyBboxPatch((0.5, 8), 2, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='gray', facecolor='lightgray', linewidth=1)
    ax.add_patch(rect)
    ax.text(1.5, 8.3, '(B*16, 3, 224, 224)', ha='center', va='center', fontsize=8)
    
    # Seta 2
    ax.arrow(1.5, 8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # 3. CNN (ResNet-34)
    rect = patches.FancyBboxPatch((0.2, 5.5), 2.6, 1.5, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='darkgreen', facecolor='lightgreen', linewidth=3)
    ax.add_patch(rect)
    ax.text(1.5, 6.7, 'CNN: ResNet-34', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.5, 6.4, '(Pré-treinado ImageNet)', ha='center', va='center', fontsize=8)
    ax.text(1.5, 6.1, '21.3M parâmetros', ha='center', va='center', fontsize=8)
    ax.text(1.5, 5.8, 'Output: 512-d features', ha='center', va='center', fontsize=8)
    
    # Seta 3
    ax.arrow(1.5, 5.5, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # 4. RESHAPE
    rect = patches.FancyBboxPatch((0.5, 4), 2, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='gray', facecolor='lightgray', linewidth=1)
    ax.add_patch(rect)
    ax.text(1.5, 4.3, '(B, 16, 512)', ha='center', va='center', fontsize=8)
    
    # Seta 4
    ax.arrow(1.5, 4, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # 5. BiLSTM
    rect = patches.FancyBboxPatch((0.2, 1.8), 2.6, 1.4, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='darkred', facecolor='lightcoral', linewidth=3)
    ax.add_patch(rect)
    ax.text(1.5, 2.9, 'BiLSTM', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.5, 2.6, '2 camadas, 256 units', ha='center', va='center', fontsize=8)
    ax.text(1.5, 2.3, '3.2M parâmetros', ha='center', va='center', fontsize=8)
    ax.text(1.5, 2.0, 'Dropout: 0.3', ha='center', va='center', fontsize=8)
    
    # Seta 5
    ax.arrow(1.5, 1.8, 0, -0.8, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # 6. FC + Sigmoid
    rect = patches.FancyBboxPatch((0.5, 0.2), 2, 0.8, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.5, 0.8, 'FC + Sigmoid', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(1.5, 0.5, '512 → 1', ha='center', va='center', fontsize=8)
    ax.text(1.5, 0.3, 'Dropout: 0.3', ha='center', va='center', fontsize=8)
    
    # OUTPUT (lado direito)
    rect = patches.FancyBboxPatch((7.5, 5.5), 2, 1, 
                                   boxstyle="round,pad=0.1", 
                                   edgecolor='orange', facecolor='lightyellow', linewidth=2)
    ax.add_patch(rect)
    ax.text(8.5, 6.3, 'OUTPUT', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8.5, 6.0, '(B, 1)', ha='center', va='center', fontsize=8)
    ax.text(8.5, 5.7, 'P(fake) ∈ [0, 1]', ha='center', va='center', fontsize=8)
    
    # Seta do FC para OUTPUT
    ax.annotate('', xy=(7.5, 6), xytext=(2.5, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Informações laterais
    info_y = 9.5
    ax.text(5, info_y, 'ESPECIFICAÇÕES:', ha='left', va='top', 
            fontsize=10, fontweight='bold')
    
    specs = [
        'Total de parâmetros: 24.4M',
        'Parâmetros treináveis: 24.4M',
        '',
        'Input: 16 frames/vídeo',
        'Frame size: 224×224 RGB',
        'Normalização: ImageNet stats',
        '',
        'Tempo de inferência:',
        '  • 1 vídeo: ~0.006s',
        '  • Batch 8: ~0.0006s/vídeo'
    ]
    
    for i, spec in enumerate(specs):
        ax.text(5, info_y - 0.4 - i*0.35, spec, ha='left', va='top', fontsize=8)
    
    # Pipeline flow
    flow_y = 4.5
    ax.text(5, flow_y, 'PIPELINE:', ha='left', va='top', 
            fontsize=10, fontweight='bold')
    
    pipeline = [
        '1. Extract 16 frames',
        '2. Detect faces (MTCNN)',
        '3. Resize to 224×224',
        '4. Normalize (ImageNet)',
        '5. CNN features (512-d)',
        '6. LSTM temporal (512-d)',
        '7. Classify (sigmoid)',
        '8. Output probability'
    ]
    
    for i, step in enumerate(pipeline):
        ax.text(5, flow_y - 0.4 - i*0.35, step, ha='left', va='top', fontsize=8)
    
    plt.tight_layout()
    
    # Salvar
    output_path = 'outputs/figures/model_architecture.png'
    ensure_dir('outputs/figures')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Diagrama salvo: {output_path}")
    plt.close()


def main():
    """Função principal."""
    print("=" * 60)
    print("CRIAÇÃO DE DIAGRAMA DA ARQUITETURA")
    print("=" * 60)
    print()
    
    create_architecture_diagram()
    
    print()
    print("✓ Visualização criada com sucesso!")
    print()


if __name__ == '__main__':
    main()
