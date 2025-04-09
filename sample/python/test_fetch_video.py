from tdl import llm

# 测试视频处理性能并获取时间统计
result = llm.test_fetch_video_ts(
    video_path="H264- 1080P 30Fps - 1 Min - Full Motion - Campus.mp4",
    desired_fps=2.0,
    desired_nframes=0,
    max_video_sec=0
)

# 打印结果
print("视频处理性能统计:")
print(f"视频打开时间: {result['duration_video_open']:.2f} ms")
print(f"读取时间: {result['duration_read']:.2f} ms")
print(f"处理时间: {result['duration_process']:.2f} ms")
print(f"视频关闭时间: {result['duration_video_close']:.2f} ms")
print(f"读取的帧数: {result['read_frames']}")
print(f"处理的帧数: {result['process_frames']}")