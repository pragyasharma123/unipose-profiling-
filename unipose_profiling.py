import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.wasp import build_wasp
from model.modules.decoder import build_decoder
from model.modules.backbone import build_backbone
import json
from torch.profiler import profile, record_function, ProfilerActivity

class unipose(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=15,
                 sync_bn=True, freeze_bn=False, stride=8):
        super(unipose, self).__init__()
        self.stride = stride

        BatchNorm = nn.BatchNorm2d
        self.num_classes = num_classes

        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.wasp = build_wasp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)  # Adjusted to remove dataset dependency

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)
        if self.stride != 8:
            x = F.interpolate(x, size=(input.size()[2:]), mode='bilinear', align_corners=True)
        return x


def benchmark_model(model, device, batch_size=32, num_repeats=100):
    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, 3, 368, 368).to(device)
    model.eval()

    # Set up CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(num_repeats):
                _ = model(input_tensor)

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # convert milliseconds

    throughput = num_repeats / (elapsed_time/1000) 
    print(f"Throughput: {throughput:.2f} inferences per second")

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    return throughput


def main():
    batch_size = 32
    num_repeats = 100
    output_json = None  # Replace with your desired output path if needed

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = unipose(num_classes=15).to(device)
    model.load_state_dict(torch.load('/home/ps332/myViT/UniPose/UniPose_MPII.pth', map_location=device))

    # Benchmark the model
    throughput = benchmark_model(model, device, batch_size, num_repeats)

    output_data_dict = {
        "batch_size": batch_size,
        "num_repeats": num_repeats,
        "throughput": throughput
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(output_data_dict, f, indent=4)
        print(f"Results saved to {output_json}")


if __name__ == '__main__':
    main()
