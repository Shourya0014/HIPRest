import time
import torch
import yaml
from basicsr.models.archs import define_network
from torchvision import transforms
from PIL import Image

def load_config(cfg_path=r'options\test\GoPro\HINet-GoPro.yml'):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def build_model(cfg, device):
    net = define_network(cfg['network_g']).to(device)
    ckpt = torch.load(cfg['path']['pretrain_network_g'], map_location=device)
    net.load_state_dict(ckpt if 'params' not in ckpt else ckpt['params'], strict=True)
    net.eval()
    return net

def load_sample_image(path, device, size=None):
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    to_tensor = transforms.ToTensor()
    return to_tensor(img).unsqueeze(0).to(device)

def measure_fps(model, input_tensor, runs=100, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()

    avg_time = (t1 - t0) / runs
    fps = 1.0 / avg_time
    return avg_time, fps

def main():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['num_gpu'] > 0 else 'cpu')
    model = build_model(cfg, device)

    sample_path = r'datasets\test\Gopro50\input\GOPR0384_11_00-000001.png'
    inp = load_sample_image(sample_path, device, size=(256,256))  # or None

    avg_time, fps = measure_fps(model, inp, runs=100, warmup=10)
    print(f'✅ Average inference time: {avg_time*1000:.1f} ms')
    print(f'✅ FPS: {fps:.2f} frames per second')

if __name__ == '__main__':
    main()
