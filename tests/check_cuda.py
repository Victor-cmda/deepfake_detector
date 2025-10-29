import torch

print("="*60)
print("VERIFICAÇÃO DE CUDA")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Teste rápido
    print("\nTestando operação CUDA...")
    x = torch.rand(100, 100).cuda()
    y = x * 2
    print(f"✓ Operação CUDA executada com sucesso!")
else:
    print("❌ CUDA NÃO DISPONÍVEL!")
    
print("="*60)
