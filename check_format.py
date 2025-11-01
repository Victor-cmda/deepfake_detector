"""Verifica formato dos arquivos WildDeepfake"""

from pathlib import Path

# Verificar primeiro arquivo
file_path = Path(r"E:\datasets\wilddeepfake_cache\datasets--xingjunm--WildDeepfake\snapshots\f3835aaf281dd9f8d79b51c4e02f050d3f7af0b4\deepfake_in_the_wild\fake_train\1.tar.gz")

with open(file_path, 'rb') as f:
    header = f.read(4)
    print(f"Primeiros 4 bytes: {header.hex()}")
    
    # Verificar formato
    if header[:2] == b'\x1f\x8b':
        print("✅ É um arquivo GZIP válido")
    elif header[:2] == b'us' or header == b'ustar':
        print("✅ É um arquivo TAR (sem compressão)")
    elif header[:4] == b'PK\x03\x04':
        print("✅ É um arquivo ZIP")
    else:
        print(f"❓ Formato desconhecido")
        print(f"   Header: {header}")
