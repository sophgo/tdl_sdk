import argparse
import cv2
import numpy as np
import os

"""
用法示例：
1) 将 RGB 图像 转换为 YUV422：
   python convert.py --mode rgb2yuv --width 0 --height 0 \
                    --input input.png --output output.yuv
   （width/height=0 表示从图像文件中自动获取分辨率）

2) 将 YUV422 (Planar) 数据 转换为 RGB 并输出为 PNG：
   python convert.py --mode yuv2rgb --width 640 --height 480 \
                    --input input.yuv --output output.png
"""

def rgb_to_yuv422_planar(rgb_image: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    将 RGB 图像转换为 YUV422 (Planar) 格式，返回三个平面 (Y, U, V)。
    - rgb_image: shape (H, W, 3)，通道顺序为 RGB
    - 返回:
        Y: shape (H, W)
        U_422: shape (H, W//2)
        V_422: shape (H, W//2)
    """
    # 1. RGB -> YUV444
    yuv444 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    Y_444 = yuv444[:, :, 0]
    U_444 = yuv444[:, :, 1]
    V_444 = yuv444[:, :, 2]

    # 2. 水平抽样 -> YUV422
    U_422 = U_444[:, ::2]
    V_422 = V_444[:, ::2]
    return Y_444, U_422, V_422


def save_yuv422_planar(Y: np.ndarray, U: np.ndarray, V: np.ndarray, filename: str):
    """
    将 Y, U, V 三个平面 (Planar) 顺序写入文件，形成 YUV422 Planar 格式的二进制数据。
    - Y: shape (H, W)
    - U: shape (H, W//2)
    - V: shape (H, W//2)
    """
    yuv_data = np.concatenate([Y.flatten(), U.flatten(), V.flatten()]).astype(np.uint8)
    yuv_data.tofile(filename)
    print(f"[INFO] YUV422 planar data saved to: {filename}")


def load_yuv422_planar(filepath: str, width: int, height: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    从二进制文件中读取 YUV422 (Planar) 格式数据，并返回 (Y, U, V) 三个平面。
    - filepath: .yuv 文件路径
    - width, height: 图像分辨率
    - 返回:
      Y: shape (H, W)
      U: shape (H, W//2)
      V: shape (H, W//2)
    """
    # 计算各平面大小
    frame_size_y = width * height          # Y 分量
    frame_size_u = (width // 2) * height   # U 分量
    frame_size_v = (width // 2) * height   # V 分量

    # 读取二进制数据
    with open(filepath, 'rb') as f:
        raw = f.read()

    if len(raw) < (frame_size_y + frame_size_u + frame_size_v):
        raise ValueError("[ERROR] 文件大小不匹配或数据不足.")

    # 将数据分割为 Y, U, V
    Y_raw = raw[0 : frame_size_y]
    U_raw = raw[frame_size_y : frame_size_y + frame_size_u]
    V_raw = raw[frame_size_y + frame_size_u : frame_size_y + frame_size_u + frame_size_v]

    # reshape
    Y_plane = np.frombuffer(Y_raw, dtype=np.uint8).reshape((height, width))
    U_plane = np.frombuffer(U_raw, dtype=np.uint8).reshape((height, width // 2))
    V_plane = np.frombuffer(V_raw, dtype=np.uint8).reshape((height, width // 2))

    return Y_plane, U_plane, V_plane


def yuv422_planar_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    将 YUV422 (Planar) 格式转换为 RGB 图像 (H, W, 3)。
    - Y shape = (H, W)
    - U shape = (H, W//2)
    - V shape = (H, W//2)
    先将 U, V 在水平方向上插值回到 W，再与 Y 组合成 YUV444，再转 RGB。
    """
    h, w = Y.shape[:2]

    # 1. 水平上 2 倍插值 -> YUV444
    #   np.repeat(axis=1) 把 (H, W//2) 变成 (H, W)
    U_444 = np.repeat(U, 2, axis=1)
    V_444 = np.repeat(V, 2, axis=1)

    # 2. 合并成 YUV444: shape (H, W, 3)
    yuv444 = np.stack((Y, U_444, V_444), axis=-1)

    # 3. 用 OpenCV 转回 RGB
    rgb_image = cv2.cvtColor(yuv444, cv2.COLOR_YUV2RGB)
    return rgb_image


def main():
    parser = argparse.ArgumentParser(description="RGB <-> YUV422 Planar Conversion")
    parser.add_argument('--mode', type=str, required=True,
                        choices=["rgb2yuv", "yuv2rgb"],
                        help="转换模式：rgb2yuv 或 yuv2rgb")
    parser.add_argument('--width', type=int, default=1920,
                        help="图像宽度 (对 yuv2rgb 有效); 若为 0 则从输入图像获取")
    parser.add_argument('--height', type=int, default=1080,
                        help="图像高度 (对 yuv2rgb 有效); 若为 0 则从输入图像获取")
    parser.add_argument('--input', type=str, required=True,
                        help="输入文件路径：图片或 yuv")
    parser.add_argument('--output', type=str, required=True,
                        help="输出文件路径：yuv 或 图片")
    args = parser.parse_args()

    mode = args.mode
    w = args.width
    h = args.height
    input_path = args.input
    output_path = args.output

    if mode == "rgb2yuv":
        # 读取输入图像 (BGR)
        bgr_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise FileNotFoundError(f"Failed to load image: {input_path}")
        # 转为 RGB
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (w, h))
        # 将 RGB 转为 YUV422 Planar
        Y, U, V = rgb_to_yuv422_planar(rgb_img)

        # 保存为二进制 .yuv
        save_yuv422_planar(Y, U, V, output_path)

    elif mode == "yuv2rgb":
        if w <= 0 or h <= 0:
            raise ValueError("[ERROR] 请指定有效的 width/height 参数.")

        # 从二进制 yuv 文件加载 YUV422 planar
        Y_plane, U_plane, V_plane = load_yuv422_planar(input_path, w, h)

        # 转成 RGB 图像
        rgb_img = yuv422_planar_to_rgb(Y_plane, U_plane, V_plane)
        # 转成 BGR（便于用 OpenCV 保存）
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        # 保存为 PNG/JPG
        cv2.imwrite(output_path, bgr_img)
        print(f"[INFO] RGB image saved to: {output_path}")

if __name__ == "__main__":
    main()