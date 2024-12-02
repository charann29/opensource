# Next-Gen AI Technologies Faculty Development Programme
## Malla Reddy University, Hyderabad
### December 6-7, 2024

## Event Details
- **Venue**: Malla Reddy University, Maisammaguda, Kompally, Hyderabad - 500100
- **Focus Areas**: 
  - Explainable AI (XAI)
  - Vector Databases & LlamaIndex
  - Large Language Model Integration
  - AI Application Development
  - System Integration Techniques

## Installation Requirements

### 1. Development Environment

#### Visual Studio Code
- Download: [https://code.visualstudio.com/](https://code.visualstudio.com/)
- Required Extensions:
  - [Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
  - [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
  - [GitLens](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)

#### Python Installation
- Download Python 3.10.12: [https://www.python.org/downloads/release/python-31012/](https://www.python.org/downloads/release/python-31012/)
- Installation Steps:
  ```bash
  # Verify Python installation
  python --version
  
  # Create virtual environment
  python -m venv fdp_venv
  
  # Activate virtual environment
  # Windows:
  .\fdp_venv\Scripts\activate
  # Linux/macOS:
  source fdp_venv/bin/activate
  ```

### 2. Version Control Tools

#### GitHub Desktop
- Download: [https://desktop.github.com/](https://desktop.github.com/)
- Features needed for the workshop:
  - Repository creation
  - Commit and push changes
  - Branch management

#### Git
- Download: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Basic configuration:
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your.email@example.com"
  ```

### 3. Required Python Libraries

#### Core AI/ML Libraries
```bash
# Update pip first
pip install --upgrade pip

# Install core libraries
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0
```

#### Deep Learning Frameworks
```bash
# TensorFlow
pip install tensorflow==2.13.0

# PyTorch (CPU version)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

#### Vector Databases & LLM Tools
```bash
# Vector database tools
pip install faiss-cpu==1.7.4
pip install llama-index==0.8.4

# OpenAI integration
pip install openai==0.28.1
```

#### Jupyter Environment
```bash
pip install notebook==7.0.3
pip install ipykernel==6.25.1
```

### 4. API Development Tools

#### Postman
- Download: [https://www.postman.com/downloads/](https://www.postman.com/downloads/)
- Features needed:
  - API testing
  - Environment setup
  - Request collection management

### 5. Cloud Service Accounts

#### OpenAI Account Setup
1. Sign up at [https://platform.openai.com/](https://platform.openai.com/)
2. Create API key
3. Set environment variable:
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/macOS
export OPENAI_API_KEY="your-api-key-here"
```

### 6. Hardware Requirements
- Minimum:
  - 8GB RAM
  - Intel i5/AMD Ryzen 5 or equivalent
  - 10GB free storage
- Recommended:
  - 16GB RAM
  - Intel i7/AMD Ryzen 7 or better
  - Dedicated GPU (NVIDIA preferred)
  - 20GB free storage

## Verification Script

Save and run this script to verify installations:
```python
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import faiss
import openai
from llama_index import GPTVectorStoreIndex

def check_versions():
    versions = {
        'Python': sys.version,
        'NumPy': np.__version__,
        'Pandas': pd.__version__,
        'TensorFlow': tf.__version__,
        'PyTorch': torch.__version__,
        'FAISS': faiss.__version__,
        'OpenAI': openai.__version__
    }
    
    print("Installation Verification Report")
    print("===============================")
    for package, version in versions.items():
        print(f"{package}: {version}")

if __name__ == "__main__":
    check_versions()
```

## Additional Resources

### Documentation Links
- [Python Documentation](https://docs.python.org/3.10/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [LlamaIndex Documentation](https://gpt-index.readthedocs.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

### Learning Materials
- [Explainable AI Techniques](https://christophm.github.io/interpretable-ml-book/)
- [Vector Database Fundamentals](https://www.pinecone.io/learn/vector-database/)
- [Large Language Models Guide](https://www.deeplearning.ai/short-courses/large-language-models/)

### Troubleshooting Common Issues

#### Python/pip Issues
```bash
# If pip is not recognized
python -m ensurepip --default-pip

# If installation fails
pip install package_name --no-cache-dir
```

#### Virtual Environment Problems
```bash
# If venv creation fails
python -m pip install --user virtualenv
python -m virtualenv fdp_venv

# If activation fails on Windows
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### GPU Setup (Optional)
- NVIDIA drivers: [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)
- CUDA Toolkit: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

## Contact Information
- Event Convener: Dr. Thayyaba Khatoon, Dean AIML
- Co-Convener: Dr. Gifta Jerith
- Website: [www.mallareddyuniversity.ac.in](http://www.mallareddyuniversity.ac.in)

## Pre-Workshop Checklist
- [ ] Python 3.10.12 installed and verified
- [ ] VS Code installed with extensions
- [ ] Git and GitHub Desktop configured
- [ ] Virtual environment created and activated
- [ ] All required libraries installed
- [ ] OpenAI API key obtained and configured
- [ ] Postman installed
- [ ] Verification script run successfully
- [ ] Documentation bookmarked for reference
