LLVM_ROOT="/home/wangzhe/zhe/llvm-project/release"
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/wangzhe/.local/envs/zhe_ipex/lib:$LD_LIBRARY_PATH
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
# -DBUILD_MODULE_TYPE=CPU -DBUILD_WITH_XPU=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=/home/wangzhe/zhe/intel-extension-for-pytorch/build/Release/packages/intel_extension_for_pytorch -DCMAKE_PREFIX_PATH=/home/wangzhe/.local/envs/zhe_ipex/lib/python3.9/site-packages/torch/share/cmake -DCMAKE_PROJECT_VERSION=2.0.100 -DIPEX_INSTALL_LIBDIR=/home/wangzhe/zhe/intel-extension-for-pytorch/build/Release/packages/intel_extension_for_pytorch/lib -DIPEX_PROJ_NAME=intel_extension_for_pytorch -DLIBIPEX_GITREV=25b721274 -DLIBIPEX_VERSION=2.0.100+git25b7212 -DPYTHON_EXECUTABLE=/home/wangzhe/.local/envs/zhe_ipex/bin/python -DPYTHON_INCLUDE_DIR=/home/wangzhe/.local/envs/zhe_ipex/include/python3.9 -DPYTHON_PLATFORM_INFO=Linux-5.16.0-rc1-intel-next-00543-g5867b0a2a125-x86_64-with-glibc2.28 -DUSE_LLVM=/home/wangzhe/zhe/llvm-project/release
