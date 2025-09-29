import sys
import os
import numpy as np
from tdl import nn, image

# 尝试导入BytePairEncoder，如果失败则禁用文本功能
try:
    from tdl.utils import BytePairEncoder
    TOKENIZER_AVAILABLE = True
    print("BytePairEncoder imported successfully")
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: BytePairEncoder not available, text features disabled")

def check_model_id_name(model_id_name):
    """检查模型ID是否支持"""
    supported_models = [
        "FEATURE_CLIP_IMG",
        "FEATURE_MOBILECLIP2_IMG", 
        "FEATURE_BMFACE_R34",
        "FEATURE_BMFACE_R50",
        "FEATURE_CVIFACE"
    ]
    
    if model_id_name not in supported_models:
        print(f"model_id_name: {model_id_name} not supported")
        sys.exit(1)
    
    # 如果是文本模型但分词器不可用
    if model_id_name == "FEATURE_CLIP_TEXT" and not TOKENIZER_AVAILABLE:
        print("Error: FEATURE_CLIP_TEXT requires BytePairEncoder which is not available")
        print("Please use C++ version for text feature extraction")
        sys.exit(1)

def get_input_data(model_id_name, input_path):
    """根据模型类型准备输入数据"""
    input_datas = []
    
    if model_id_name == "FEATURE_CLIP_TEXT":
        if not TOKENIZER_AVAILABLE:
            print("BytePairEncoder not available")
            sys.exit(1)
            
        # 处理文本输入
        txt_dir = input_path.rstrip('/')
        encoder_file = os.path.join(txt_dir, "encoder.txt")
        bpe_file = os.path.join(txt_dir, "vocab.txt") 
        input_file = os.path.join(txt_dir, "input.txt")
        
        # 检查文件是否存在
        for file_path in [encoder_file, bpe_file, input_file]:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                sys.exit(1)
        
        try:
            # 使用C++的BytePairEncoder
            bpe = BytePairEncoder(encoder_file, bpe_file)
            tokens = bpe.tokenizer_bpe(input_file)
            
            # 处理每个token序列
            for i, token_seq in enumerate(tokens):
                # 输出格式与C++版本保持一致
                print(" ".join(map(str, token_seq)))
                print(f"Current token index i: {i}")
                
                # 创建文本图像（77维token序列）
                # 修复：使用numpy数组创建图像
                token_array = np.array(token_seq[:77], dtype=np.int32)
                if len(token_array) < 77:
                    padded_tokens = np.zeros(77, dtype=np.int32)
                    padded_tokens[:len(token_array)] = token_array
                    token_array = padded_tokens
                
                # 将一维数组重塑为二维(1, 77)
                token_array = token_array.reshape(1, 77)
                
                # 使用from_numpy方法创建图像
                text_image = image.Image.from_numpy(token_array, image.ImageFormat.GRAY)
                
                input_datas.append(text_image)
        except Exception as e:
            print(f"Text processing failed: {e}")
            sys.exit(1)
    else:
        # 图像处理
        if not os.path.exists(input_path):
            print("Failed to load images")
            sys.exit(1)
            
        img = image.read(input_path)
        if img is None:
            print("Failed to load images")
            sys.exit(1)
        input_datas.append(img)
    
    return input_datas

def get_model_type_from_name(model_id_name):
    """将模型名称映射到ModelType"""
    model_type_map = {
        "FEATURE_CLIP_IMG": nn.ModelType.FEATURE_CLIP_IMG,
        "FEATURE_MOBILECLIP2_IMG": nn.ModelType.FEATURE_MOBILECLIP2_IMG,
        "FEATURE_BMFACE_R34": nn.ModelType.FEATURE_BMFACE_R34,
        "FEATURE_BMFACE_R50": nn.ModelType.FEATURE_BMFACE_R50,
        "FEATURE_CVIFACE": nn.ModelType.FEATURE_CVIFACE,
    }
    return model_type_map.get(model_id_name)

def save_feature_to_file(feature_array, data_type_str, index):
    """保存特征到文件"""
    filename = f"feature_{data_type_str}_{index}.bin"
    
    with open(filename, 'wb') as f:
        f.write(feature_array.tobytes())
    
    print(f"The feature file have been saved as feature_{data_type_str}_{index}.bin")
    return filename

def process_feature(feature_data, index):
    """处理单个特征输出"""
    print(f"Processing feature {index}, type: {type(feature_data)}")
    
    # 处理numpy标量类型
    if isinstance(feature_data, (np.int8, np.uint8, np.float32, np.int32, np.float64)):
        print(f"Single scalar value: {feature_data} (type: {type(feature_data)})")
        return False  # 标量无法作为特征保存
    
    # 处理numpy数组
    elif isinstance(feature_data, np.ndarray):
        # 检查是否是标量数组
        if feature_data.ndim == 0:
            print(f"Scalar array: {feature_data.item()} (dtype: {feature_data.dtype})")
            return False
        
        feature_array = feature_data
        print(f"feature size: {feature_array.size}")
        print(f"feature shape: {feature_array.shape}")
        print(f"feature dtype: {feature_array.dtype}")
        
        # 根据数据类型保存文件（与C++版本保持一致）
        if feature_array.dtype == np.int8:
            save_feature_to_file(feature_array, "int8", index)
        elif feature_array.dtype == np.uint8:
            save_feature_to_file(feature_array, "uint8", index)
        elif feature_array.dtype == np.float32:
            save_feature_to_file(feature_array, "fp32", index)
        else:
            # 转换为float32作为默认类型
            print(f"Converting {feature_array.dtype} to float32")
            feature_array_fp32 = feature_array.astype(np.float32)
            save_feature_to_file(feature_array_fp32, "fp32", index)
        return True
    
    # 处理字典格式
    elif isinstance(feature_data, dict):
        print(f"Feature is dict with keys: {feature_data.keys()}")
        if 'embedding' in feature_data:
            embedding = feature_data['embedding']
            if isinstance(embedding, np.ndarray):
                print(f"feature size: {embedding.size}")
                
                if embedding.dtype == np.int8:
                    save_feature_to_file(embedding, "int8", index)
                elif embedding.dtype == np.uint8:
                    save_feature_to_file(embedding, "uint8", index)
                elif embedding.dtype == np.float32:
                    save_feature_to_file(embedding, "fp32", index)
                else:
                    embedding_fp32 = embedding.astype(np.float32)
                    save_feature_to_file(embedding_fp32, "fp32", index)
                return True
    
    # 处理列表格式
    elif isinstance(feature_data, list):
        print(f"Feature is list with {len(feature_data)} elements")
        if len(feature_data) > 0:
            print(f"First element type: {type(feature_data[0])}")
        return False
    
    else:
        print(f"Unknown feature format: {type(feature_data)}")
        return False

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <model_id_name> <model_dir> <image_path>")
        print(f"Usage: {sys.argv[0]} <model_id_name> <model_dir> <txt_dir>")
        return -1
        
    model_id_name = sys.argv[1]
    model_dir = sys.argv[2]
    input_path = sys.argv[3]
    
    print(f"Model ID: {model_id_name}")
    print(f"Model directory: {model_dir}")
    print(f"Input path: {input_path}")
    
    # 检查模型ID
    check_model_id_name(model_id_name)
    
    # 获取输入数据
    input_datas = get_input_data(model_id_name, input_path)
    print(f"Input data prepared: {len(input_datas)} items")
    
    # 获取模型类型
    model_type = get_model_type_from_name(model_id_name)
    if model_type is None:
        print(f"Failed to load model")
        return -1
    
    # 加载模型
    try:
        print("Loading model...")
        model = nn.get_model_from_dir(model_type, model_dir)
        if model is None:
            print("Failed to load model")
            return -1
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return -1
    
    # 执行推理
    try:
        feature_count = 0
        # 处理每个输入数据
        for i, input_data in enumerate(input_datas):
            print(f"\nStarting inference for input {i}")
            out_features = model.inference(input_data)
            
            print(f"Inference result type: {type(out_features)}")
            
            # 处理输出特征
            if isinstance(out_features, list):
                print(f"Output is list with {len(out_features)} elements")
                
                # 检查是否所有元素都是标量（可能需要合并）
                if len(out_features) > 0:
                    first_type = type(out_features[0])
                    print(f"First element type: {first_type}")
                    
                    # 如果是标量列表，转换为数组
                    if isinstance(out_features[0], (np.int8, np.uint8, np.float32, np.int32)):
                        print("Converting scalar list to numpy array")
                        feature_array = np.array(out_features)
                        print(f"Converted array shape: {feature_array.shape}, dtype: {feature_array.dtype}")
                        
                        # 保存合并后的特征
                        if feature_array.dtype == np.int8:
                            save_feature_to_file(feature_array, "int8", feature_count)
                        elif feature_array.dtype == np.uint8:
                            save_feature_to_file(feature_array, "uint8", feature_count)
                        elif feature_array.dtype == np.float32:
                            save_feature_to_file(feature_array, "fp32", feature_count)
                        else:
                            feature_array_fp32 = feature_array.astype(np.float32)
                            save_feature_to_file(feature_array_fp32, "fp32", feature_count)
                        feature_count += 1
                    else:
                        # 处理每个特征
                        for j, feature in enumerate(out_features):
                            if process_feature(feature, feature_count):
                                feature_count += 1
            else:
                if process_feature(out_features, feature_count):
                    feature_count += 1
        
        print(f"\nTotal features saved: {feature_count}")
                
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return -1
    
    return 0

if __name__ == "__main__":
    result = main()
    sys.exit(result)