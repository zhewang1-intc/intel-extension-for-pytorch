#!/bin/bash
set -x
set -e

VER_LLVM="llvmorg-13.0.0"
VER_PYTORCH=""
VER_TORCHVISION=""
VER_TORCHAUDIO=""
VER_IPEX="v2.0.100+cpu"

# Check existance of required Linux commands
for CMD in gcc g++ python git nproc; do
    command -v ${CMD} || (
        echo "Error: Command \"${CMD}\" not found."
        exit 4
    )
done
echo "You are using GCC: $(gcc --version | grep gcc)"

MAX_JOBS_VAR=$(nproc)
if [ ! -z "${MAX_JOBS}" ]; then
    MAX_JOBS_VAR=${MAX_JOBS}
fi

# Save current directory path
BASEFOLDER=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd ${BASEFOLDER}

ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# Compile individual component
#  LLVM
cd llvm-project

LLVM_ROOT="$(pwd)/release"
# ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
#  Intel® Extension for PyTorch*
cd ../intel-extension-for-pytorch
python -m pip install -r requirements.txt
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
# python setup.py clean
python setup.py bdist_wheel 2>&1 | tee build.log
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
unset USE_LLVM
python -m pip install --force-reinstall dist/*.whl

# Sanity Test
cd ..
python -c "import torch; import torchvision; import torchaudio; import intel_extension_for_pytorch as ipex; print(f'torch_cxx11_abi:     {torch._C._GLIBCXX_USE_CXX11_ABI}'); print(f'torch_version:       {torch.__version__}'); print(f'torchvision_version: {torchvision.__version__}'); print(f'torchaudio_version:  {torchaudio.__version__}'); print(f'ipex_version:        {ipex.__version__}');"
