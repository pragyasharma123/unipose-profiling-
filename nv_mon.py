import nvidia_smi
import time
import pandas as pd
from multiprocessing import Process, Manager


class GPUMonitor:
    def __init__(self, device_id):
        nvidia_smi.nvmlInit()
        self.device_id = device_id
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)

    def step(self, records, start_time):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.device_id)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        power = nvidia_smi.nvmlDeviceGetPowerUsage(handle)
        # max_power = nvidia_smi.nvmlDeviceGetEnforcedPowerLimit(handle)
        temp = nvidia_smi.nvmlDeviceGetTemperature(handle, nvidia_smi.NVML_TEMPERATURE_GPU)
        sm_clock = nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_SM)
        mem_clock = nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_MEM)
        # pcie_rx = nvidia_smi.nvmlDeviceGetPcieThroughput(handle, nvidia_smi.NVML_PCIE_UTIL_RX_BYTES)
        # pcie_tx = nvidia_smi.nvmlDeviceGetPcieThroughput(handle, nvidia_smi.NVML_PCIE_UTIL_TX_BYTES)
        # performance_state = nvidia_smi.nvmlDeviceGetPerformanceState(handle)
        # voltage = nvidia_smi.nvmlDeviceGetPowerUsage(handle)

        data_dict = {
            'time': time.time() - start_time,
            'utilization': util.gpu,
            'memory_used': mem.used,
            'memory_total': mem.total,
            'power': power,
            'temperature': temp,
            'sm_clock': sm_clock,
            'mem_clock': mem_clock,
        }
        records.append(data_dict)

    def monitor(self, records, start_time):
        nvidia_smi.nvmlInit()
        while True:
            self.step(records, start_time)

    def start_monitoring(self):
        manager = Manager()
        records = manager.list()
        start_time = time.time()
        self.process = Process(target=self.monitor, args=(records, start_time))
        self.process.start()
        return records

    def stop_monitoring(self):
        self.process.terminate()
        self.process.join()

    def get_data(self, records):
        return list(records)

# Example of using the GPUMonitor class
if __name__ == "__main__":
    gpu_monitor = GPUMonitor(0)  # Monitor GPU at index 0
    records = gpu_monitor.start_monitoring()

    # Simulate some other operations while monitoring
    time.sleep(5)  # Monitor for 10 seconds

    gpu_monitor.stop_monitoring()
    print(records)
