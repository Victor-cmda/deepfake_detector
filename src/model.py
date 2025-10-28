"""
Arquitetura do modelo DeepfakeDetector.
CNN (ResNet-34) + BiLSTM para classificação temporal de vídeos.
"""

import torch
import torch.nn as nn
from torchvision import models
import time


class DeepfakeDetector(nn.Module):
    """
    Modelo de detecção de deepfakes com arquitetura híbrida CNN-LSTM.
    
    Arquitetura:
    1. CNN: ResNet-34 pré-treinado (extração de features espaciais)
    2. BiLSTM: 2 camadas bidirecionais com 256 unidades (modelagem temporal)
    3. Classificador: FC + Sigmoid (classificação binária)
    
    Input: (batch_size, num_frames, channels, height, width)
    Output: (batch_size, 1) - probabilidade de ser deepfake
    """
    
    def __init__(self, num_frames=16, lstm_hidden=256, lstm_layers=2, dropout=0.3, pretrained=True):
        """
        Inicializa o modelo DeepfakeDetector.
        
        Args:
            num_frames (int): Número de frames por vídeo
            lstm_hidden (int): Número de unidades LSTM por direção
            lstm_layers (int): Número de camadas LSTM
            dropout (float): Taxa de dropout
            pretrained (bool): Usar ResNet-34 pré-treinado
        """
        super(DeepfakeDetector, self).__init__()
        
        self.num_frames = num_frames
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        
        # 1. CNN: ResNet-34 como extrator de features
        resnet = models.resnet34(pretrained=pretrained)
        
        # Remover camada de classificação final (FC)
        # Manter apenas as camadas convolucionais
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension do ResNet-34
        self.cnn_output_size = 512
        
        # 2. BiLSTM: Modelagem temporal
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # 3. Classificador final
        # BiLSTM output: lstm_hidden * 2 (bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Inicializar pesos da FC
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor (batch_size, num_frames, C, H, W)
            
        Returns:
            torch.Tensor: Probabilidades de deepfake (batch_size, 1)
        """
        batch_size, num_frames, c, h, w = x.size()
        
        # 1. Extrair features com CNN para cada frame
        # Reshape: (batch_size * num_frames, C, H, W)
        x = x.view(batch_size * num_frames, c, h, w)
        
        # CNN forward
        cnn_features = self.cnn(x)  # (batch_size * num_frames, 512, 1, 1)
        
        # Flatten: (batch_size * num_frames, 512)
        cnn_features = cnn_features.view(batch_size * num_frames, -1)
        
        # Reshape para sequência: (batch_size, num_frames, 512)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        
        # 2. LSTM para modelagem temporal
        lstm_out, _ = self.lstm(cnn_features)  # (batch_size, num_frames, lstm_hidden*2)
        
        # Pegar output do último frame
        lstm_out = lstm_out[:, -1, :]  # (batch_size, lstm_hidden*2)
        
        # 3. Classificação
        x = self.dropout(lstm_out)
        x = self.fc(x)  # (batch_size, 1)
        x = self.sigmoid(x)  # (batch_size, 1)
        
        return x
    
    def get_num_params(self):
        """
        Retorna o número total de parâmetros do modelo.
        
        Returns:
            dict: Dicionário com contagem de parâmetros
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Contar parâmetros por módulo
        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        fc_params = sum(p.numel() for p in self.fc.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'cnn': cnn_params,
            'lstm': lstm_params,
            'fc': fc_params
        }
    
    def freeze_cnn(self):
        """
        Congela os parâmetros da CNN (ResNet-34).
        Útil para fine-tuning apenas do LSTM.
        """
        for param in self.cnn.parameters():
            param.requires_grad = False
        print("✓ CNN congelada (parâmetros não treináveis)")
    
    def unfreeze_cnn(self):
        """
        Descongela os parâmetros da CNN.
        """
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("✓ CNN descongelada (parâmetros treináveis)")
    
    def get_feature_extractor(self):
        """
        Retorna apenas a parte CNN do modelo.
        Útil para extração de features.
        
        Returns:
            nn.Module: Módulo CNN
        """
        return self.cnn


def create_model(num_frames=16, lstm_hidden=256, lstm_layers=2, dropout=0.3, pretrained=True, device='cpu'):
    """
    Cria e inicializa o modelo DeepfakeDetector.
    
    Args:
        num_frames (int): Número de frames por vídeo
        lstm_hidden (int): Unidades LSTM por direção
        lstm_layers (int): Número de camadas LSTM
        dropout (float): Taxa de dropout
        pretrained (bool): Usar pesos pré-treinados do ResNet-34
        device (str): Device ('cpu', 'cuda', 'mps')
        
    Returns:
        DeepfakeDetector: Modelo inicializado
    """
    model = DeepfakeDetector(
        num_frames=num_frames,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
        pretrained=pretrained
    )
    
    model = model.to(device)
    
    # Estatísticas do modelo
    params = model.get_num_params()
    
    print("\n" + "=" * 60)
    print("MODELO DEEPFAKE DETECTOR")
    print("=" * 60)
    print(f"\nArquitetura:")
    print(f"  - CNN: ResNet-34 {'(pré-treinado)' if pretrained else '(aleatório)'}")
    print(f"  - LSTM: BiLSTM {lstm_layers} camadas, {lstm_hidden} unidades")
    print(f"  - Dropout: {dropout}")
    print(f"  - Frames: {num_frames}")
    print(f"\nParâmetros:")
    print(f"  - Total: {params['total']:,}")
    print(f"  - Treináveis: {params['trainable']:,}")
    print(f"  - CNN: {params['cnn']:,}")
    print(f"  - LSTM: {params['lstm']:,}")
    print(f"  - FC: {params['fc']:,}")
    print(f"\nDevice: {device}")
    print("=" * 60 + "\n")
    
    return model


def test_model_forward(model, batch_size=2, num_frames=16, device='cpu'):
    """
    Testa o forward pass do modelo com dados dummy.
    
    Args:
        model (DeepfakeDetector): Modelo a testar
        batch_size (int): Tamanho do batch
        num_frames (int): Número de frames
        device (str): Device
        
    Returns:
        dict: Estatísticas do teste
    """
    print(f"Testando forward pass...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Num frames: {num_frames}")
    print(f"  - Input shape: ({batch_size}, {num_frames}, 3, 224, 224)")
    
    # Criar input dummy
    dummy_input = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
    
    # Forward pass com medição de tempo
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(dummy_input)
        forward_time = time.time() - start_time
    
    print(f"\n✓ Forward pass concluído!")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  - Tempo: {forward_time:.4f}s")
    print(f"  - Tempo por amostra: {forward_time/batch_size:.4f}s")
    
    return {
        'output_shape': output.shape,
        'forward_time': forward_time,
        'time_per_sample': forward_time / batch_size
    }


def save_model(model, path, optimizer=None, epoch=None, metrics=None):
    """
    Salva o modelo e informações de treinamento.
    
    Args:
        model (DeepfakeDetector): Modelo a salvar
        path (str): Caminho do arquivo
        optimizer (torch.optim.Optimizer): Otimizador (opcional)
        epoch (int): Época atual (opcional)
        metrics (dict): Métricas (opcional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_frames': model.num_frames,
            'lstm_hidden': model.lstm_hidden,
            'lstm_layers': model.lstm_layers
        }
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, path)
    print(f"✓ Modelo salvo: {path}")


def load_model(path, device='cpu'):
    """
    Carrega o modelo de um checkpoint.
    
    Args:
        path (str): Caminho do checkpoint
        device (str): Device
        
    Returns:
        DeepfakeDetector: Modelo carregado
    """
    checkpoint = torch.load(path, map_location=device)
    
    config = checkpoint['model_config']
    model = DeepfakeDetector(
        num_frames=config['num_frames'],
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Modelo carregado: {path}")
    
    if 'epoch' in checkpoint:
        print(f"  - Época: {checkpoint['epoch']}")
    
    if 'metrics' in checkpoint:
        print(f"  - Métricas: {checkpoint['metrics']}")
    
    return model, checkpoint

