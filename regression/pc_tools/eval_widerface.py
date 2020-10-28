import os
import subprocess
import argparse

def evaluation(pred_folder, mlir_python_path):
    wider_eval_tool = os.path.join(mlir_python_path, 'cvi_toolkit', 'dataset_util', 'widerface', 'wider_eval_tools')
    folder_name = os.path.basename(pred_folder)

    cmd = 'cp -r {} {};'.format(pred_folder, wider_eval_tool)
    cmd += 'pushd {};'.format(wider_eval_tool)
    cmd += 'octave --no-window-system --eval \"wider_eval(\'{}\', \'{}\')\" 2>&1;'.format(folder_name, 'face')
    cmd += 'rm -rf {};'.format(folder_name)
    cmd += 'popd;'
    print("exec:{}".format(cmd))
    subprocess.call(cmd, shell=True, executable='/bin/bash')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', metavar='result_dir', help='Result directory')
    parser.add_argument('mlir_python', metavar='mlir_path', help='MLIR python path')
    args = parser.parse_args()

    args.result_dir = args.result_dir.rstrip('/')
    evaluation(args.result_dir, args.mlir_python)
