# Third-party code

## Transolver++

`dgp/models/transolver.py` contains `Model`, `Transolver_plus_block` and `Physics_Attention_1D_Eidetic` from
`models/Transolver_plus.py` of https://github.com/thuml/Transolver_plus (Luo et al., *Transolver++: An Accurate
Neural Solver for PDEs on Million-Scale Geometries*, ICML 2025), distributed under the MIT License:

```
MIT License

Copyright (c) 2025 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GINO

`dgp/models/gino.py` wraps `neuralop.models.GINO` from the `neuraloperator` package
(https://github.com/neuraloperator/neuraloperator, MIT License), installed as a dependency; no code is copied.
