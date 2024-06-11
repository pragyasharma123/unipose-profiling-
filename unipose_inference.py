import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.wasp import build_wasp
from model.modules.decoder import build_decoder
from model.modules.backbone import build_backbone
import time
import argparse
from nv_mon import GPUMonitor
import json

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


def benchmark_model(model, device, args):
    # Create a dummy input tensor
    input_tensor = torch.randn(args.batch_size, 3, 368, 368).to(device)
    model.eval()

    # Set up CUDA events for precise timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    gpu_monitor = GPUMonitor(0)  # Monitor GPU at index 0
    records = gpu_monitor.start_monitoring()
    torch.cuda.synchronize()
    start_event.record()
    '''
    # Warm up
    for _ in range(10):
        _ = model(input_tensor)
    '''
    # Measure throughput
    for _ in range(args.num_repeats):
        _ = model(input_tensor)

    end_event.record()
    torch.cuda.synchronize()
    gpu_monitor.stop_monitoring()
    records = gpu_monitor.get_data(records)
    elapsed_time = start_event.elapsed_time(end_event)  # convert milliseconds

    throughput = args.num_repeats / (elapsed_time/1000) 
    print(f"Throughput: {throughput:.2f} inferences per second")

    return throughput, records

def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference throughput.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the inference.")
    parser.add_argument("--num_repeats", type=int, default=100,
                        help="Number of times to repeat the inference for throughput measurement.")
    parser.add_argument("--output_json", type=str, default=None,
                        help="The file path for where to output a json of the benchmark results.")                    
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = unipose(num_classes=15).to(device)
    model.load_state_dict(torch.load('/home/ps332/myViT/UniPose/UniPose_MPII.pth', map_location=device))

    # Benchmark the model
    throughput, records = benchmark_model(model, device, args)

    output_data_dict = {
        "batch_size": args.batch_size,
        "num_repeats": args.num_repeats,
        "throughput": throughput,
        "gpu_records": records
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(output_data_dict, f, indent=4)
        print(f"Results saved to {args.output_json}")


if __name__ == '__main__':
    main()
