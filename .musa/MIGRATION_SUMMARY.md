# mmdetection v2.26.0 MUSA Migration Summary

## Migration Date
2026-04-10

## Migration Type
SimplePorting (Pure Python project, no CUDA source files)

## Changed Files (16 files)
1. `configs/_base_/default_runtime.py` - NCCL → MCCL backend
2. `mmdet/apis/inference.py` - device='cuda:0' → 'musa:0', .is_cuda → .is_musa
3. `mmdet/apis/test.py` - device='cuda' → 'musa'
4. `mmdet/apis/train.py` - device='cuda' → 'musa', torch.cuda.manual_seed_all → torch.musa
5. `mmdet/core/bbox/iou_calculators/iou2d_calculator.py` - .is_cuda → .is_musa
6. `mmdet/core/bbox/samplers/random_sampler.py` - torch.cuda.is_available → torch.musa
7. `mmdet/core/bbox/samplers/score_hlr_sampler.py` - torch.cuda.is_available → torch.musa
8. `mmdet/core/data_structures/general_data.py` - added .musa() method, .cuda() alias
9. `mmdet/core/export/model_wrappers.py` - CUDAExecutionProvider → MUSA, torch.cuda.device → torch.musa.device
10. `mmdet/models/losses/focal_loss.py` - torch.cuda.is_available → torch.musa.is_available
11. `mmdet/utils/contextmanagers.py` - torch.cuda.Stream/Event → torch.musa
12. `mmdet/utils/memory.py` - AvoidCUDAOOM → AvoidMUSAOOM, torch.cuda.empty_cache → torch.musa
13. `mmdet/utils/profiling.py` - torch.cuda.is_available/Event → torch.musa
14. `mmdet/utils/util_distribution.py` - added MUSA device support, default device='musa'
15. `setup.py` - CUDAExtension → MUSAExtension, make_cuda_ext → make_musa_ext
16. `tools/deployment/pytorch2onnx.py` - .cuda() → .musa()

## API Mapping Applied
| CUDA API | MUSA API |
|----------|----------|
| torch.cuda.is_available() | torch.musa.is_available() |
| torch.cuda.current_device() | torch.musa.current_device() |
| torch.cuda.current_stream() | torch.musa.current_stream() |
| torch.cuda.Event() | torch.musa.Event() |
| torch.cuda.Stream() | torch.musa.Stream() |
| torch.cuda.empty_cache() | torch.musa.empty_cache() |
| torch.cuda.manual_seed_all() | torch.musa.manual_seed_all() |
| torch.cuda.device() | torch.musa.device() |
| .cuda() | .musa() |
| .is_cuda | .is_musa |
| device='cuda' | device='musa' |
| CUDAExtension | MUSAExtension |
| nccl backend | mccl backend |
| torch.backends.cudnn | torch.backends.mudnn |

## Dependencies
- Requires mmcv-musa (CUDA ops from mmcv must be musified separately)
- Requires torch_musa

## Backward Compatibility
- Added alias `AvoidCUDAOOM = AvoidOOM()` for backward compatibility
- Added alias `make_cuda_ext = make_musa_ext` for backward compatibility
- `.cuda()` method kept as alias to `.musa()` in general_data.py

## Testing Required
1. Smoke test: `python tools/test.py` with a simple config
2. Unit tests: `pytest tests/`
3. Training test: `python tools/train.py`
4. Inference test: `python demo/inference_demo.py`

## Notes
- No CUDA source files (.cu) in this project - all CUDA ops come from mmcv
- mmcv must be musified before mmdet can work on MUSA