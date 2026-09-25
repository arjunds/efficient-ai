import pynvml, time, subprocess
pynvml.nvmlInit()
print("driver", pynvml.nvmlSystemGetDriverVersion(), "nvml", pynvml.nvmlSystemGetNVMLVersion())
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print(pynvml.nvmlDeviceGetName(h))
print("enforced limit W", pynvml.nvmlDeviceGetEnforcedPowerLimit(h)/1e3)
print("constraints", [x/1e3 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)])
print("default limit", pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h)/1e3)
print("max clocks sm/mem", pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM), pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM))
print("mem bus width", pynvml.nvmlDeviceGetMemoryBusWidth(h) if hasattr(pynvml,'nvmlDeviceGetMemoryBusWidth') else None)
print("energy counter mJ", pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
for name in ["NVML_FI_DEV_POWER_INSTANT","NVML_FI_DEV_POWER_AVERAGE"]:
    fid = getattr(pynvml, name, None); print(name, fid)
    if fid is not None:
        try:
            v = pynvml.nvmlDeviceGetFieldValues(h, [fid])[0]
            print("  ret", v.nvmlReturn, "val", v.value.uiVal, v.value.ullVal)
        except Exception as e: print("  err", e)
# sample rate test of energy counter
e0=pynvml.nvmlDeviceGetTotalEnergyConsumption(h); vals=[]; t0=time.time()
while time.time()-t0<1.0:
    vals.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(h)); time.sleep(0.002)
import itertools
ch=sum(1 for a,b in zip(vals,vals[1:]) if a!=b); print("energy counter changes in 1s:", ch, "of", len(vals))
ps=[]; t0=time.time()
while time.time()-t0<1.0:
    ps.append(pynvml.nvmlDeviceGetPowerUsage(h)); time.sleep(0.002)
print("power reading changes in 1s:", sum(1 for a,b in zip(ps,ps[1:]) if a!=b), "of", len(ps), "mean", sum(ps)/len(ps)/1e3)
import torch
print(torch.cuda.get_device_properties(0))
print(subprocess.run(["nvidia-smi","-q","-d","POWER,CLOCK,PERFORMANCE"],capture_output=True,text=True).stdout[:4000])
