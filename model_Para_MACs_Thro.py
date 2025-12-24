import torch
import time
from thop import profile, clever_format
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.RhythmFormer import RhythmFormer
from neural_methods.model.RhythmMamba import RhythmMamba
from neural_methods.model.LQC2F import Label_Quantizer, C2F_model

def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model_name, model, input_size, is_physformer=False, num_iterations=1000):
    """Measures the inference time of the model."""
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    if is_physformer:
        gra_sharp = torch.tensor(0.7).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            if is_physformer:
                _ = model(dummy_input, gra_sharp)
            else:
                _ = model(dummy_input)
    
    # Start measurement
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            if is_physformer:
                _ = model(dummy_input, gra_sharp)
            else:
                _ = model(dummy_input)
            torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    # Calculate throughput (Kfps)
    throughput = (160 / avg_time)
    
    return avg_time, throughput

def analyze_model(model, input_size, model_name, is_physformer=False):
    """Analyzes the computational cost of the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Generate dummy input
    dummy_input = torch.randn(input_size).to(device)
    if is_physformer:
        gra_sharp = torch.tensor(0.7).to(device)
        macs, params = profile(model, inputs=(dummy_input, gra_sharp))
        macs = macs / 160
    else:
        macs, params = profile(model, inputs=(dummy_input,))
        macs = macs / 160

    # Convert MACs and params to M units
    macs = f"{macs / 1_000_000:.2f}M"
    params = f"{params / 1_000_000:.2f}M"
    
    # Measure inference time and throughput
    inference_time, throughput = measure_inference_time(model_name, model, input_size, is_physformer)
    
    print(f"\n===== {model_name} Model Analysis Results =====")
    print(f"Input size: {input_size}")
    print(f"Total parameters: {params}")
    print(f"MACs per frame: {macs}")
    print(f"Throughput: {(throughput/1000):.2f} Kfps")

def main():    
    # DeepPhys
    # img_size is the original image size, target_size is the downsampled image size

    deepphys = DeepPhys(in_channels=3, img_size=128, target_size=128)
    analyze_model(deepphys, (160, 6, 128, 128), "DeepPhys")

    # from torchinfo import summary
    # summary(deepphys, input_size=(1, 6, 96, 96))
    
    # PhysNet
    physnet = PhysNet_padding_Encoder_Decoder_MAX(frames=160)
    analyze_model(physnet, (1, 3, 160, 128, 128), "PhysNet")
    
    # TSCAN
    tscan = TSCAN(in_channels=3, img_size=128, frame_depth=160)
    analyze_model(tscan, (160, 6, 128, 128), "TSCAN")
    
    # PhysFormer
    physformer = ViT_ST_ST_Compact3_TDC_gra_sharp(
        name='PhysFormer',
        patches=(4, 4, 4),  
        dim=96, 
        ff_dim=144,  
        num_heads=4,  
        num_layers=12,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        frame=160,
        theta=0.7,
        image_size=(160, 128, 128)
    )
    analyze_model(physformer, (1, 3, 160, 128, 128), "PhysFormer", is_physformer=True)
    
    # EfficientPhys
    # img_size is the original image size, target_size is the downsampled image size
    efficientphys = EfficientPhys(in_channels=3, frame_depth=160, img_size=128, target_size=128)
    analyze_model(efficientphys, (161, 3, 128, 128), "EfficientPhys")
    
    # Rhythmformer
    rhythmformer = RhythmFormer()
    analyze_model(rhythmformer, (1, 160, 3, 128, 128), "RhythmFormer")

    # RhythmMamba
    rhythmmamba = RhythmMamba()
    analyze_model(rhythmmamba, (1, 160, 3, 128, 128), "RhythmMamba")

    # Q2FPhys
    q2fphys = C2F_model()
    analyze_model(q2fphys, (1, 3, 160, 128, 128), "`C2F_model`")

if __name__ == "__main__":
    main() 