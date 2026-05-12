# mmdetection v2.26.0 CUDA Analysis Summary

## Project Type
- **Pure Python Project**: No CUDA source files (.cu/.cuh)
- **Conversion Strategy**: SimplePorting (Python torch.cuda replacement)

## CUDA Dependency Chain
- **Primary CUDA Dependency**: mmcv-full >= 1.3.17
  - mmcv contains all CUDA ops (RoIPool, RoIAlign, NMS, etc.)
  - mmcv must be musified first before mmdet

## Python CUDA API Usage (33 files)
### Core torch.cuda APIs:
- `torch.cuda.is_available()` - device detection
- `torch.cuda.Stream()` - async inference
- `torch.cuda.empty_cache()` - memory management
- `device='cuda:0'` / `device='cuda'` - device string literals
- `.cuda()` - tensor/device transfer
- `.is_cuda` - tensor device check

### Distributed Training:
- PyTorch DDP (torch.distributed)
- NCCL backend usage via mmcv

## Files Requiring Modification
### API Surface (High Priority):
1. `mmdet/apis/inference.py` - device='cuda:0' default, .is_cuda check
2. `mmdet/apis/train.py` - distributed training setup
3. `mmdet/apis/test.py` - device='cuda' usage
4. `mmdet/utils/memory.py` - torch.cuda.empty_cache()
5. `mmdet/utils/profiling.py` - CUDA profiling
6. `setup.py` - torch.cuda.is_available(), CUDAExtension

### Test Files (Medium Priority):
- tests/test_runtime/test_async.py
- tests/test_runtime/test_fp16.py
- tests/test_utils/test_memory.py
- tests/test_models/test_forward.py

### Tools (Lower Priority):
- tools/analysis_tools/benchmark.py
- tools/deployment/*.py

## Build System
- setup.py uses CUDAExtension but has no actual CUDA extensions defined
- ext_modules=[] (empty list)
- BuildExtension imported but not actively used

## Minimum CUDA Version
- Estimated: CUDA 10.2+ (based on mmcv requirements)

## Migration Priority
1. mmcv-full (must be musified first)
2. mmdet Python torch.cuda replacements