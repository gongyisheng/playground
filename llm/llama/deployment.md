# deploy on mac M1
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python

# run 
python3 -m llama2_cpp.server --config_file XXXX.yaml