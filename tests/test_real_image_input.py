# SPDX-License-Identifier: Apache-2.0
import torch
from PIL import Image
import numpy as np
from vllm.multimodal.processing import MultiModalInputProcessor

def test_real_image_processing():
    print("--- LitevLLM Real Image Input Test ---")
    
    # 1. 模拟用户输入：创建一个真实的 PIL 图像
    print("Creating a synthetic PIL image (224x224)...")
    raw_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    
    # 2. 模拟请求格式
    prompt = "Describe this image."
    mm_data = {"image": raw_image}
    
    # 3. 初始化处理器
    processor = MultiModalInputProcessor(registry=None)
    
    # 4. 执行处理
    print("Processing multimodal inputs...")
    try:
        processed_inputs = processor.process_inputs(prompt, mm_data)
        
        # 5. 验证结果
        print("\n--- Verification ---")
        print(f"Prompt: {processed_inputs.prompt}")
        
        if processed_inputs.multi_modal_data and "image" in processed_inputs.multi_modal_data:
            pixel_values = processed_inputs.multi_modal_data["image"]
            print(f"Success: Found 'image' tensor in multi_modal_data.")
            print(f"Tensor Shape:  {pixel_values.shape}")
            print(f"Tensor Device: {pixel_values.device}")
            
            # 最终断言
            assert isinstance(pixel_values, torch.Tensor)
            assert pixel_values.is_cuda
            print("\nFinal Result: Image to Tensor conversion PASSED.")
        else:
            print("Failure: 'image' data not found in output.")
            
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise e

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_real_image_processing()
    else:
        print("CUDA not available, skipping real image test.")
