"""
Teste final da interface corrigida.
"""

import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

print("=" * 70)
print("LANÇANDO INTERFACE GRADIO CORRIGIDA")
print("=" * 70)
print()

print("Correções aplicadas:")
print("  ✅ Upload via gr.File (ao invés de gr.Video)")
print("  ✅ Conversão automática de vídeo para formato compatível")
print("  ✅ Galeria de frames processados (visualização)")
print("  ✅ Modo de avaliação do modelo corrigido (cudnn RNN)")
print("  ✅ Validação robusta de inputs")
print()

# Verificar modelo
model_path = 'models/model_best.pt'
if not os.path.exists(model_path):
    print(f"❌ ERRO: Modelo não encontrado em {model_path}")
    sys.exit(1)

print(f"✓ Modelo encontrado: {model_path}")
print()

# Importar e lançar
try:
    from src.interface import launch_interface
    
    print("Iniciando servidor Gradio...")
    print("Acesse: http://localhost:7861")
    print()
    print("=" * 70)
    print()
    
    launch_interface(share=False, server_port=7861)
    
except KeyboardInterrupt:
    print("\n\nServidor encerrado pelo usuário.")
except Exception as e:
    print(f"\n❌ ERRO: {e}")
    import traceback
    traceback.print_exc()
