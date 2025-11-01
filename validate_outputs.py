"""
Script para validar que todos os outputs necessários para o TCC foram gerados corretamente.
"""

from pathlib import Path
import pandas as pd

def validate_outputs():
    """Valida todos os outputs necessários conforme instructions.json"""
    
    print("\n" + "="*70)
    print("VALIDAÇÃO DE OUTPUTS PARA O TCC")
    print("="*70 + "\n")
    
    all_ok = True
    
    # Outputs esperados conforme instructions.json
    expected_outputs = {
        'models/model_best.pt': 'Modelo treinado',
        'outputs/metrics_train.csv': 'Métricas de treino',
        'outputs/metrics_cross.csv': 'Métricas cross-dataset',
        'outputs/figures/training_curves.png': 'Curvas de treinamento',
        'outputs/figures/f1_by_dataset.png': 'F1 por dataset',
        'outputs/figures/confusion_matrix.png': 'Matriz de confusão',
        'outputs/figures/gradcam_examples.png': 'Exemplos Grad-CAM',
        'outputs/reports/interface_log.csv': 'Log da interface',
        'outputs/reports/run_report.md': 'Relatório técnico',
        'outputs/reports/table_metrics.csv': 'Tabela de métricas',
        'outputs/reports/robustness.csv': 'Teste de robustez',
    }
    
    print("📋 Verificando arquivos obrigatórios:\n")
    
    for file_path, description in expected_outputs.items():
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            if size > 0:
                size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
                print(f"  ✅ {description:30s} - {file_path} ({size_str})")
            else:
                print(f"  ⚠️  {description:30s} - {file_path} (VAZIO!)")
                all_ok = False
        else:
            print(f"  ❌ {description:30s} - {file_path} (NÃO ENCONTRADO)")
            all_ok = False
    
    print("\n" + "-"*70 + "\n")
    
    # Validar conteúdo das métricas
    print("📊 Validando conteúdo das métricas:\n")
    
    # Métricas de treino
    if Path('outputs/metrics_train.csv').exists():
        df_train = pd.read_csv('outputs/metrics_train.csv')
        print(f"  ✅ metrics_train.csv: {len(df_train)} épocas")
        print(f"     - Melhor Val AUC: {df_train['val_auc'].max():.4f} (época {df_train['val_auc'].idxmax() + 1})")
        print(f"     - Melhor Val F1: {df_train['val_f1'].max():.4f} (época {df_train['val_f1'].idxmax() + 1})")
        print(f"     - Train Loss final: {df_train.iloc[-1]['train_loss']:.4f}")
    else:
        print("  ❌ metrics_train.csv não encontrado")
        all_ok = False
    
    # Métricas cross-dataset
    if Path('outputs/metrics_cross.csv').exists():
        df_cross = pd.read_csv('outputs/metrics_cross.csv')
        df_cross_valid = df_cross[df_cross['total_samples'] > 100]
        print(f"\n  ✅ metrics_cross.csv: {len(df_cross_valid)} datasets válidos")
        for _, row in df_cross_valid.iterrows():
            print(f"     - {row['dataset']}: AUC {row['auc']:.4f}, F1 {row['f1']:.4f} ({int(row['total_samples'])} amostras)")
    else:
        print("  ❌ metrics_cross.csv não encontrado")
        all_ok = False
    
    # Tabela de métricas
    if Path('outputs/reports/table_metrics.csv').exists():
        df_metrics = pd.read_csv('outputs/reports/table_metrics.csv')
        print(f"\n  ✅ table_metrics.csv: {len(df_metrics)} métricas consolidadas")
    else:
        print("  ❌ table_metrics.csv não encontrado")
        all_ok = False
    
    print("\n" + "-"*70 + "\n")
    
    # Verificar heatmaps
    heatmaps_dir = Path('outputs/heatmaps')
    if heatmaps_dir.exists():
        heatmaps = list(heatmaps_dir.glob('*.png'))
        print(f"📸 Heatmaps Grad-CAM: {len(heatmaps)} exemplos mantidos\n")
    else:
        print("⚠️  Diretório heatmaps não encontrado\n")
    
    print("="*70)
    
    if all_ok:
        print("✅ VALIDAÇÃO COMPLETA: TODOS OS OUTPUTS ESTÃO OK!")
        print("="*70)
        print("\n🎓 Arquivos prontos para uso no TCC:")
        print("\n📊 FIGURAS (outputs/figures/):")
        print("   1. training_curves.png - Curvas de treinamento")
        print("   2. f1_by_dataset.png - Comparação F1/AUC")
        print("   3. confusion_matrix.png - Matrizes de confusão")
        print("   4. gradcam_examples.png - Mapas de atenção")
        print("\n📄 RELATÓRIOS (outputs/reports/):")
        print("   1. table_metrics.csv - Métricas consolidadas")
        print("   2. run_report.md - Relatório técnico completo")
        print("\n📈 MÉTRICAS BRUTAS:")
        print("   1. outputs/metrics_train.csv - Histórico de treino")
        print("   2. outputs/metrics_cross.csv - Cross-dataset evaluation")
        print("\n📚 DOCUMENTAÇÃO:")
        print("   - OUTPUTS_TCC_REFERENCIA.md - Guia completo com textos")
        print("\n")
    else:
        print("❌ VALIDAÇÃO FALHOU: Alguns outputs estão faltando!")
        print("="*70)
        print("\nExecute novamente o script de regeneração:")
        print("  python clean_and_regenerate.py")
        print("\n")
    
    return all_ok


if __name__ == "__main__":
    import sys
    success = validate_outputs()
    sys.exit(0 if success else 1)
