import yaml

def parse_yaml_to_cmake(yaml_path, cmake_output_path):
    # 加载 YAML 配置
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config['ENABLE_SLIM'] == 0:

        # 生成 CMake 变量文件
        with open(cmake_output_path, 'w', encoding='utf-8') as f:
            f.write(f'set(ENABLE_SLIM OFF CACHE BOOL "Auto-generated from build_config.yaml")\n')
        return

    del config['ENABLE_SLIM']
    
    module_list = []
    enable_features = set() 
    disable_features = set() 

    for module in config['MODULES']:

        for k, v in module.items():
            if v == 1:
                enable_features.add(k)
                module_list.append(k.replace('ENABLE_', ''))
            else:
                disable_features.add(k)


    for module in config.keys():

        if module == 'MODULES':
            continue

        if module not in module_list:
            continue

        for config_item in config[module]:
            assert len(config_item.keys()) == 1
            config_key = list(config_item.keys())[0]

            if config_item[config_key] == 1:

                enable_features.add(config_key)

            else:
                disable_features.add(config_key)
    
    # 生成 CMake 变量文件
    with open(cmake_output_path, 'w', encoding='utf-8') as f:
        f.write(f'set(ENABLE_SLIM ON CACHE BOOL "Auto-generated from build_config.yaml")\n')
        for feature in disable_features:
            f.write(f'set({feature} OFF CACHE BOOL "Auto-generated from build_config.yaml")\n')
            tmp_fea = feature.replace('ENABLE_', 'DISABLE_')
            f.write(f'add_definitions(-D{tmp_fea})\n\n')

        for feature in enable_features:
            f.write(f'set({feature} ON CACHE BOOL "Auto-generated from build_config.yaml")\n')
            f.write(f'add_definitions(-D{feature})\n\n')
        



if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python parse_build_config.py <input.yaml> <output.cmake>")
        sys.exit(1)
    parse_yaml_to_cmake(sys.argv[1], sys.argv[2])


