#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import os
from grpc_tools import protoc

def compile_protos():
    grpc_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(grpc_path))
    
    print(f"Compiling vllm_engine.proto in {grpc_path}...")
    
    # We need to include the project root in the path so that 
    # imports like 'vllm/grpc/...' work if needed
    protoc.main((
        '',
        f'-I{project_root}',
        f'--python_out={project_root}',
        f'--grpc_python_out={project_root}',
        f'--pyi_out={project_root}',
        os.path.join(grpc_path, 'vllm_engine.proto'),
    ))
    print("Compilation successful.")

if __name__ == '__main__':
    compile_protos()
