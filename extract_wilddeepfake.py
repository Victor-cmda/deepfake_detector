"""
Script para extrair WildDeepfake de arquivos .tar.gz
"""

import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil

def extract_wilddeepfake():
    """Extrai WildDeepfake dos arquivos .tar.gz"""
    source_base = Path(r"E:\datasets\wilddeepfake_cache\datasets--xingjunm--WildDeepfake\snapshots\f3835aaf281dd9f8d79b51c4e02f050d3f7af0b4\deepfake_in_the_wild")
    target_base = Path(r"E:\deepfake_detector\data\WildDeepfake")
    
    # Criar diretórios de destino
    fake_dir = target_base / "videos_fake"
    real_dir = target_base / "videos_real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXTRAINDO WILDDEEPFAKE")
    print("=" * 60)
    print("⚠️  Processo pode demorar ~30 minutos...")
    print()
    
    # Mapear origem -> destino
    mappings = [
        (source_base / "fake_train", fake_dir, "FAKE"),
        (source_base / "fake_test", fake_dir, "FAKE"),
        (source_base / "real_train", real_dir, "REAL"),
        (source_base / "real_test", real_dir, "REAL"),
    ]
    
    stats = {"fake": 0, "real": 0, "errors": 0}
    
    for source_dir, target_dir, label in mappings:
        if not source_dir.exists():
            print(f"⚠️  {source_dir} não existe!")
            continue
        
        tar_files = list(source_dir.glob("*.tar.gz"))
        print(f"📁 {source_dir.name}: {len(tar_files)} arquivos .tar.gz ({label})")
        
        for tar_path in tqdm(tar_files, desc=f"Extraindo {label}"):
            try:
                # Nome único para o vídeo
                video_name = f"{source_dir.name}_{tar_path.stem}.mp4"
                target_path = target_dir / video_name
                
                # Pular se já existe
                if target_path.exists():
                    if label == "FAKE":
                        stats["fake"] += 1
                    else:
                        stats["real"] += 1
                    continue
                
                # Extrair tar (sem compressão gzip, apenas tar puro)
                with tarfile.open(tar_path, "r:") as tar:
                    # Procurar por arquivo de vídeo
                    video_found = False
                    for member in tar.getmembers():
                        if member.name.endswith(('.mp4', '.avi', '.mkv', '.MP4', '.AVI', '.MKV')):
                            # Extrair para temp
                            tar.extract(member, target_dir)
                            extracted_path = target_dir / member.name
                            
                            # Renomear para o nome padronizado
                            if extracted_path.exists():
                                shutil.move(str(extracted_path), str(target_path))
                                video_found = True
                                
                                if label == "FAKE":
                                    stats["fake"] += 1
                                else:
                                    stats["real"] += 1
                                break
                    
                    if not video_found:
                        print(f"\n⚠️  Nenhum vídeo encontrado em {tar_path.name}")
                        stats["errors"] += 1
                        
            except Exception as e:
                print(f"\n❌ Erro ao extrair {tar_path.name}: {e}")
                stats["errors"] += 1
                continue
    
    print()
    print("=" * 60)
    print("✅ WildDeepfake extraído:")
    print(f"   - FAKE: {stats['fake']} vídeos em {fake_dir}")
    print(f"   - REAL: {stats['real']} vídeos em {real_dir}")
    print(f"   - Erros: {stats['errors']}")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    print("🚀 Iniciando extração do WildDeepfake...")
    print()
    
    try:
        stats = extract_wilddeepfake()
        print("\n✅ Extração concluída com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro durante extração: {e}")
        import traceback
        traceback.print_exc()
