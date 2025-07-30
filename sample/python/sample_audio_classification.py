import sys
import os
import numpy as np
from tdl import nn, image

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python sample_audio_classification.py <model_id_name> <model_path> <bin_data_path>")
        sys.exit(1)

    model_id_name = sys.argv[1]
    model_path = sys.argv[2]
    bin_data_path = sys.argv[3]

    if not os.path.exists(bin_data_path):
        print(f"Binary file does not exist: {bin_data_path}")
        sys.exit(1)

    buffer = np.fromfile(bin_data_path, dtype=np.uint8)
    print("Buffer shape:", buffer.shape)

    frame_size = buffer.size

    buffer = buffer.reshape((1, frame_size, 1))  # 变成 (1, N, 1) 的三维
    bin_img = image.from_numpy(buffer, format=image.ImageFormat.GRAY)

    model_id = getattr(nn.ModelType, model_id_name)
    classifier = nn.get_model(model_id, model_path)

    result = classifier.inference(bin_img)

    output = result[0]
    print("score:", output["score"])
    print("class_id:", output["class_id"])


# input: python sample_audio_classification.py <model_id_name> <model_path> <bin_data_path>
# output: score: <score_value> class_id: <class_id_value>