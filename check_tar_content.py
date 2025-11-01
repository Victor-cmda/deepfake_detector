"""Verifica conteúdo de um arquivo TAR do WildDeepfake"""

import tarfile
from pathlib import Path

tar_path = Path(r"E:\datasets\wilddeepfake_cache\datasets--xingjunm--WildDeepfake\snapshots\f3835aaf281dd9f8d79b51c4e02f050d3f7af0b4\deepfake_in_the_wild\fake_train\1.tar.gz")

print(f"Verificando: {tar_path.name}\n")

try:
    with tarfile.open(tar_path, "r:") as tar:
        members = tar.getmembers()
        print(f"Total de membros: {len(members)}\n")
        
        for i, member in enumerate(members[:20]):  # Primeiros 20
            print(f"{i+1}. {member.name} ({member.size} bytes) - {'DIR' if member.isdir() else 'FILE'}")
            
        if len(members) > 20:
            print(f"\n... e mais {len(members) - 20} membros")
            
        # Procurar por vídeos
        videos = [m for m in members if m.name.endswith(('.mp4', '.avi', '.mkv', '.MP4', '.AVI', '.MKV'))]
        print(f"\n📹 Vídeos encontrados: {len(videos)}")
        for video in videos[:5]:
            print(f"   - {video.name} ({video.size / 1024 / 1024:.2f} MB)")
            
except Exception as e:
    print(f"Erro: {e}")
