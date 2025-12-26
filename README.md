## Requirements
You can install the required packages with the following command:

### Nvidia
```bash
conda create -name pangu-dev python=3.12
conda activate pangu-dev
pip install -r requirements.txt
```

### Ascend

Assume the vllm==0.9.2 and vllm-ascend==0.9.2rc1 are already installed, then:
```bash
pip install -r requirements-ascend.txt
```

## Acknowledgement
The codebase is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).
