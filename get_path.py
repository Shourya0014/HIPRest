import torch

pth_path = r"experiments\GoPro-HINet\models\net_g_ema_latest.pth"
checkpoint = torch.load(pth_path, map_location='cpu')

print("Top-level keys:", checkpoint.keys())
