import torch
import torch.nn as nn

class WeatherSimulator(nn.Module):
    def __init__(self, mode="rain"):
        super().__init__()
        self.mode = mode
    def forward(self, x):
        if self.mode == "rain": return x + torch.randn_like(x) * 0.1
        elif self.mode == "fog": return torch.clamp(x * 0.5 + 0.3, 0, 1)
        return x

if __name__ == "__main__":
    sim = WeatherSimulator(mode="fog")
    sample = torch.rand(1, 3, 256, 256)
    output = sim(sample)
    print(f"Weather-simulated output shape: {output.shape}")