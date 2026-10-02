Device `cpu`, torch 2.14.1+cpu, 4 CPUs (Intel(R) Xeon(R) Processor), 4 torch threads.
CUDA available at measurement time: False.
Patch and frame figures are the mean ± sample standard deviation of the recorded repeats.

| Model | Params | FLOPs/pixel | Patch images/s | ms/frame |
| --- | ---: | ---: | ---: | ---: |
| A | 24953 | 49770 | 630.08 ± 55.68 (64×64, batch 16) | 506.38 ± 16.27 (1280×720) |
| B | 20327 | 40381 | 466.08 ± 72.85 (64×64, batch 16) | 811.68 ± 78.50 (1280×720) |
| C | 24607 | 23307 | 1270.07 ± 231.42 (64×64, batch 16) | 401.13 ± 7.91 (1280×720) |
| D | 17367 | 16455 | 1678.85 ± 135.41 (64×64, batch 16) | 492.59 ± 29.23 (1280×720) |
| E | 8277 | 9028 | 1518.19 ± 126.53 (64×64, batch 16) | 527.32 ± 28.92 (1280×720) |

ONNX graphs were checked at the export resolution. The legacy tracer does not make height and width dynamic.

| Model | Max abs diff | ONNX Runtime images/s | PyTorch images/s | Timing |
| --- | ---: | ---: | ---: | --- |
| A | 1.192e-07 | 1339.25 | 752.48 | batch 8, 64×64 |
| B | 1.192e-07 | 1150.34 | 1198.46 | batch 8, 64×64 |
| C | 2.086e-07 | 1370.02 | 1410.10 | batch 8, 64×64 |
| D | 1.192e-07 | 1822.58 | 1625.07 | batch 8, 64×64 |
| E | 5.960e-08 | 1914.56 | 1298.88 | batch 8, 64×64 |
