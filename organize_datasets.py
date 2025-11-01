"""
Script para organizar WildDeepfake e Celeb-DF nas estruturas corretas.
Garante que vídeos fake e real sejam categorizados corretamente.

NOTA: WildDeepfake está em arquivos .tar.gz e precisa ser extraído primeiro.
FaceForensics++ já está organizado em FaceForensics++_C23.
"""

import os
import shutil
import tarfile
from pathlib import Path
from tqdm import tqdm

def extract_and_organize_wilddeepfake():
    """
    Extrai e organiza WildDeepfake de arquivos .tar.gz:
    E:/datasets/wilddeepfake_cache/.../deepfake_in_the_wild/{fake_train, fake_test, real_train, real_test}/*.tar.gz
    
    Para:
    data/WildDeepfake/videos_fake/ (todos os fake_train + fake_test)
    data/WildDeepfake/videos_real/ (todos os real_train + real_test)
    """
    source_base = Path(r"E:\datasets\wilddeepfake_cache\datasets--xingjunm--WildDeepfake\snapshots\f3835aaf281dd9f8d79b51c4e02f050d3f7af0b4\deepfake_in_the_wild")
    target_base = Path(r"E:\deepfake_detector\data\WildDeepfake")
    
    # Criar diretórios de destino
    fake_dir = target_base / "videos_fake"
    real_dir = target_base / "videos_real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXTRAINDO E ORGANIZANDO WILDDEEPFAKE")
    print("=" * 60)
    print("⚠️  Arquivos em .tar.gz, processo pode levar alguns minutos...")
    
    # Mapear origem -> destino com validação
    mappings = [
        (source_base / "fake_train", fake_dir, "FAKE"),
        (source_base / "fake_test", fake_dir, "FAKE"),
        (source_base / "real_train", real_dir, "REAL"),
        (source_base / "real_test", real_dir, "REAL"),
    ]
    
    stats = {"fake": 0, "real": 0}
    
    for source_dir, target_dir, label in mappings:
        if not source_dir.exists():
            print(f"⚠️  AVISO: {source_dir} não existe!")
            continue
        
        tar_files = list(source_dir.glob("*.tar.gz"))
        print(f"\n📁 {source_dir.name}: {len(tar_files)} arquivos .tar.gz ({label})")
        
        for tar_path in tqdm(tar_files, desc=f"Extraindo {label}"):
            try:
                # Extrair tar.gz
                with tarfile.open(tar_path, "r:gz") as tar:
                    # Obter nome do vídeo (assumindo 1 vídeo por tar.gz)
                    members = tar.getmembers()
                    for member in members:
                        if member.name.endswith(('.mp4', '.avi', '.mkv')):
                            # Criar nome único: {pasta}_{numero}.mp4
                            video_name = f"{source_dir.name}_{tar_path.stem}.mp4"
                            target_path = target_dir / video_name
                            
                            # Evitar sobrescrever
                            if target_path.exists():
                                continue
                            
                            # Extrair para destino
                            member.name = video_name  # Renomear durante extração
                            tar.extract(member, target_dir)
                            
                            # Mover para o lugar correto se necessário
                            extracted_path = target_dir / member.name
                            if extracted_path != target_path:
                                shutil.move(extracted_path, target_path)
                            
                            if label == "FAKE":
                                stats["fake"] += 1
                            else:
                                stats["real"] += 1
                            break  # Apenas 1 vídeo por tar.gz
                            
            except Exception as e:
                print(f"\n⚠️  Erro ao extrair {tar_path.name}: {e}")
                continue
    
    print(f"\n✅ WildDeepfake extraído e organizado:")
    print(f"   - FAKE: {stats['fake']} vídeos em {fake_dir}")
    print(f"   - REAL: {stats['real']} vídeos em {real_dir}")
    
    return stats


def organize_faceforensicspp():
    """
    Organiza FaceForensics++ de:
    data/FaceForensics++/FaceForensics++_C23/{DeepFakeDetection, Deepfakes, Face2Face, etc.}
    
    Para:
    data/FaceForensics++/videos_fake/ (todos os deepfakes)
    data/FaceForensics++/videos_real/ (original)
    """
    source_base = Path(r"E:\deepfake_detector\data\FaceForensics++\FaceForensics++_C23")
    target_base = Path(r"E:\deepfake_detector\data\FaceForensics++")
    
    # Criar diretórios de destino
    fake_dir = target_base / "videos_fake"
    real_dir = target_base / "videos_real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ORGANIZANDO FACEFORENSICS++")
    print("=" * 60)
    
    # Mapear categorias -> destino
    fake_categories = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    real_category = "original"
    
    stats = {"fake": 0, "real": 0}
    
    # Processar vídeos FAKE
    for category in fake_categories:
        category_dir = source_base / category
        if not category_dir.exists():
            print(f"⚠️  {category} não existe!")
            continue
        
        videos = list(category_dir.rglob("*.mp4"))
        print(f"\n📁 {category}: {len(videos)} vídeos (FAKE)")
        
        for video in tqdm(videos, desc=f"Copiando FAKE ({category})"):
            # Nome único: categoria_video.mp4
            target_name = f"{category}_{video.name}"
            target_path = fake_dir / target_name
            
            if target_path.exists():
                continue
            
            shutil.copy2(video, target_path)
            stats["fake"] += 1
    
    # Processar vídeos REAL
    real_dir_source = source_base / real_category
    if real_dir_source.exists():
        videos = list(real_dir_source.rglob("*.mp4"))
        print(f"\n📁 {real_category}: {len(videos)} vídeos (REAL)")
        
        for video in tqdm(videos, desc="Copiando REAL"):
            target_path = real_dir / video.name
            
            if target_path.exists():
                continue
            
            shutil.copy2(video, target_path)
            stats["real"] += 1
    
    print(f"\n✅ FaceForensics++ organizado:")
    print(f"   - FAKE: {stats['fake']} vídeos em {fake_dir}")
    print(f"   - REAL: {stats['real']} vídeos em {real_dir}")
    
    return stats


def organize_celebdf():
    """
    Organiza Celeb-DF de:
    E:/Celeb/Celeb-synthesis/ (FAKE)
    E:/Celeb/Celeb-real/ (REAL)
    E:/Celeb/YouTube-real/ (REAL)
    
    Para:
    data/Celeb-DF-v2/videos_fake/ (todos os Celeb-synthesis)
    data/Celeb-DF-v2/videos_real/ (Celeb-real + YouTube-real)
    """
    source_base = Path(r"E:\Celeb")
    target_base = Path(r"E:\deepfake_detector\data\Celeb-DF-v2")
    
    # Criar diretórios de destino
    fake_dir = target_base / "videos_fake"
    real_dir = target_base / "videos_real"
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ORGANIZANDO CELEB-DF")
    print("=" * 60)
    
    # Mapear origem -> destino com validação
    mappings = [
        (source_base / "Celeb-synthesis", fake_dir, "FAKE"),
        (source_base / "Celeb-real", real_dir, "REAL"),
        (source_base / "YouTube-real", real_dir, "REAL"),
    ]
    
    stats = {"fake": 0, "real": 0}
    
    for source_dir, target_dir, label in mappings:
        if not source_dir.exists():
            print(f"⚠️  AVISO: {source_dir} não existe!")
            continue
        
        videos = list(source_dir.glob("*.mp4"))
        print(f"\n📁 {source_dir.name}: {len(videos)} vídeos ({label})")
        
        for video in tqdm(videos, desc=f"Copiando {label}"):
            # Preservar origem no nome para evitar conflitos
            target_name = f"{source_dir.name}_{video.name}"
            target_path = target_dir / target_name
            
            # Evitar sobrescrever
            if target_path.exists():
                print(f"⚠️  {target_name} já existe, pulando...")
                continue
            
            shutil.copy2(video, target_path)
            
            if label == "FAKE":
                stats["fake"] += 1
            else:
                stats["real"] += 1
    
    print(f"\n✅ Celeb-DF organizado:")
    print(f"   - FAKE: {stats['fake']} vídeos em {fake_dir}")
    print(f"   - REAL: {stats['real']} vídeos em {real_dir}")
    
    return stats


def verify_organization():
    """Verifica a organização final dos datasets."""
    datasets = [
        Path(r"E:\deepfake_detector\data\WildDeepfake"),
        Path(r"E:\deepfake_detector\data\Celeb-DF-v2"),
        Path(r"E:\deepfake_detector\data\FaceForensics++"),
    ]
    
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO FINAL")
    print("=" * 60)
    
    total_fake = 0
    total_real = 0
    
    for dataset_dir in datasets:
        if not dataset_dir.exists():
            print(f"⚠️  {dataset_dir.name} não existe!")
            continue
        
        fake_dir = dataset_dir / "videos_fake"
        real_dir = dataset_dir / "videos_real"
        
        fake_count = len(list(fake_dir.glob("*.mp4"))) if fake_dir.exists() else 0
        real_count = len(list(real_dir.glob("*.mp4"))) if real_dir.exists() else 0
        
        total_fake += fake_count
        total_real += real_count
        
        print(f"\n📊 {dataset_dir.name}:")
        print(f"   - FAKE: {fake_count} vídeos")
        print(f"   - REAL: {real_count} vídeos")
    
    print(f"\n" + "=" * 60)
    print(f"TOTAL: {total_fake + total_real} vídeos")
    print(f"   - FAKE: {total_fake}")
    print(f"   - REAL: {total_real}")
    print(f"   - Proporção FAKE/REAL: {total_fake/total_real:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    print("🚀 Iniciando organização dos datasets...")
    print("⚠️  Este processo pode levar alguns minutos.\n")
    
    # Organizar FaceForensics++
    try:
        ff_stats = organize_faceforensicspp()
    except Exception as e:
        print(f"❌ Erro ao organizar FaceForensics++: {e}")
        ff_stats = {"fake": 0, "real": 0}
    
    # Organizar Celeb-DF
    try:
        cd_stats = organize_celebdf()
    except Exception as e:
        print(f"❌ Erro ao organizar Celeb-DF: {e}")
        cd_stats = {"fake": 0, "real": 0}
    
    # Organizar WildDeepfake (OPCIONAL - arquivos .tar.gz, processo lento)
    print("\n" + "=" * 60)
    resposta = input("Deseja extrair WildDeepfake? (demora ~30min) [s/N]: ")
    if resposta.lower() in ['s', 'sim', 'y', 'yes']:
        try:
            wd_stats = extract_and_organize_wilddeepfake()
        except Exception as e:
            print(f"❌ Erro ao organizar WildDeepfake: {e}")
            wd_stats = {"fake": 0, "real": 0}
    else:
        print("⏭️  WildDeepfake ignorado (pode extrair depois se necessário)")
        wd_stats = {"fake": 0, "real": 0}
    
    # Verificar organização final
    verify_organization()
    
    print("\n✅ Organização concluída!")
    print("📝 Próximos passos:")
    print("   1. Verificar se os vídeos estão nas pastas corretas")
    print("   2. Processar os novos datasets com generate_splits.py")
    print("   3. Treinar o modelo com train_full.py")
