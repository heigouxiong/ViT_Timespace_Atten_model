import torch
import sys

from thop import profile, clever_format

sys.path.append("/home/ubuntu/zq_mae/ViT")

from ViT_pro_timespace_attention import SpaceTimeViT
from LSTM_train import OptimizedLSTM
from RNN_train import OptimizedRNN
from Transformer_train import CSITransformer


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    param_size = 0
    for p in model.parameters():
        param_size += p.numel() * p.element_size()

    buffer_size = 0
    for b in model.buffers():
        buffer_size += b.numel() * b.element_size()

    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def analyze_model(model, model_name, dummy_input, device):
    model = model.to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)

    try:
        macs, params_from_thop = profile(
            model,
            inputs=(dummy_input,),
            verbose=False
        )
        flops = 2 * macs

        macs_str, params_str = clever_format([macs, params_from_thop], "%.3f")
        flops_str, _ = clever_format([flops, params_from_thop], "%.3f")
    except Exception as e:
        macs_str = f"统计失败: {e}"
        flops_str = f"统计失败: {e}"
        params_str = "统计失败"

    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Total Params:      {total_params:,}")
    print(f"Trainable Params:  {trainable_params:,}")
    print(f"Model Size:        {model_size_mb:.2f} MB")
    print(f"Params (thop):     {params_str}")
    print(f"MACs:              {macs_str}")
    print(f"FLOPs:             {flops_str}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 你的输入配置
    window_size = 4
    input_dim = 2 * 256 * 128

    # dummy input: (B, T, F)
    dummy_input = torch.randn(1, window_size, input_dim).to(device)

    # 1. ViT
    vit_model = SpaceTimeViT(
        img_size=(256, 128),
        patch_size=(16, 16),
        in_chans=2,
        T=window_size,
        embed_dim=512,
        depth=4,
        num_heads=16,
        drop_path_rate=0.1
    )

    # 2. LSTM
    lstm_model = OptimizedLSTM()

    # 3. RNN
    rnn_model = OptimizedRNN()

    # 4. Transformer
    transformer_model = CSITransformer(window_size=window_size)

    models = [
        ("ViT", vit_model),
        ("LSTM", lstm_model),
        ("RNN", rnn_model),
        ("Transformer", transformer_model),
    ]

    for model_name, model in models:
        analyze_model(model, model_name, dummy_input, device)


if __name__ == "__main__":
    main()