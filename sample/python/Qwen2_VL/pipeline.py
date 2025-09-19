import time
import argparse
from PIL import Image
from tdl import llm
import json
import os
import sys
sys.path.append('/data/python-packages')
import torch
from typing import Optional, Tuple
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, PretrainedConfig, Qwen2VLConfig
def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "video" in vision_info:
            video_path = vision_info['video']
            fps = vision_info.get('fps', 1.0)
            video_input = torch.tensor(llm.fetch_video(video_path, desired_fps=fps), dtype=torch.float32)
            print("fps", fps)
            video_sample_fps = fps
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs

def get_rope_index(
        config,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

def get_position_ids(processor, config, video_path, text="Describe this video."):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 0.5,
                },
                {"type": "text", "text": "你是一个智能看家助理。"\
                                      "请按照以下要求对视频进行分析，并输出三个部分的内容，注意格式要点："\
                                      "1) 描述视频主要画面与活动，不超过 100 字，需包含人物（如有）、环境背景、主要动作等信息，但不得冗余或重复描述同一对象或细节："\
                                      "描述: <请在此处输出描述，字数不超过 100 字>"\
                                      "2) 结合视频内容，为出现的对象生成多层级标签，使用 JSON 格式，标签内容务必覆盖视频中出现的相关对象："\
                                      "标签:"\
                                      "{"\
                                      "\"一级标签\": 在[\"人\", \"宠物\", \"车\", \"家门\", \"包裹\"] 中实际出现的对象,"\
                                      "\"二级标签\": 如果出现 \"人\" 时，在 [\"男\", \"女\", \"无法判断\"] 中判断人物性别,"\
                                      "\"三级标签\": 如果出现 \"人\" 时，在 [\"老年\", \"中年\", \"青年\", \"儿童\", \"无法判断\"] 中判断人物年龄"\
                                      "}"\
                                      "3) 根据视频里对象活动是否具备明确目标进行“活动评估”："\
                                      "- 如果观察到对象在进行有明确目标的活动（如：做家务、做作业、玩手机、打电话、吃法喝水等），输出："\
                                      "活动评估: 有意义"\
                                      "- 如果对象的行动无特定目标（如：随意走动、长时间停滞等），输出："\
                                      "活动评估: 无意义"\
                                      "请严格按照上述格式依次输出以上三部分内容，并且不要添加任何多余文本。"},
            ],
        }
    ]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": video_path,
    #                 "max_pixels": 360 * 420,
    #                 "fps": 1.0,
    #             },
    #             {"type": "text", "text": "你是一个智能看家助理。"\
    #                                   "请按照以下要求对视频进行分析，注意格式要点."\
    #                                   "描述视频主要画面与活动，不超过 100 字，需包含人物（如有）、环境背景、主要动作等信息："\
    #                                   "\"描述: <请在此处输出描述，字数不超过 100 字>\""\
    #                                   "请一定按格式输出！"},
    #         ],
    #     }
    # ]
    # breakpoint()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs =process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    SEQ_LENGTH = config['max_position_embeddings']
    SEQ_LENGTH = 2048
    # SEQ_LENGTH = 6000
    # SEQ_LENGTH = self.SEQLEN
    if SEQ_LENGTH <= inputs.input_ids.shape[-1]:
        raise ValueError(
            f"The input_length must be shorter than model's seq_length (got `input_length`: {inputs.input_ids.shape[-1]}"
            f" and `seq_length`: {SEQ_LENGTH})."
        )
    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values_videos
    video_grid_thw = inputs.video_grid_thw
    input_ids_prefill = torch.zeros(1, SEQ_LENGTH).to(torch.int32)
    input_ids_prefill[:, :input_ids.shape[-1]] = input_ids
    attention_mask_prefill = torch.zeros(1, SEQ_LENGTH)
    attention_mask_prefill[:, :input_ids.shape[-1]] = inputs.attention_mask
    pretrained_config = PretrainedConfig(**config)
    # with open('./../compile/files/Qwen2-VL-2B-Instruct/config.json', 'r') as json_file:
    #     config_dict = json.load(json_file)
    #     loaded_config = Qwen2VLConfig(**config_dict)
        # print(loaded_config)
    config_dict = config
    loaded_config = Qwen2VLConfig(**config_dict)
    image_mask = (input_ids_prefill == loaded_config.video_token_id)
    true_indices = torch.nonzero(image_mask, as_tuple=True)[1]

    if true_indices.numel() > 0:
        first_true_index = true_indices[0].item()
    else:
        first_true_index = None
    
    position_ids, _ = get_rope_index(loaded_config,
        input_ids_prefill, None, video_grid_thw, attention_mask_prefill
    )

    pixel_num = true_indices.shape[-1]
    # breakpoint()
    return position_ids, inputs, first_true_index, pixel_num

class Qwen2VL():

    def __init__(self, args):
        # devid
        self.device = args.devid
        self.processor = AutoProcessor.from_pretrained(args.processor_path,
                                                       trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                       trust_remote_code=True)
        with open(args.config, 'r') as f:
            self.config = json.load(f)

        # load model
        self.model = llm.Qwen2VL()
        self.model.init(self.device, args.model_path)
        self.model.generation_mode = args.generation_mode
        # self.POSITION_IDS, _, _ = get_position_ids(processor=self.processor, config=self.config)
        self.SEQLEN = self.model.SEQLEN
        # self.ID_EOS = self.tokenizer.eos_token_id
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.video_path = args.video_path
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        # Stop Chatting with "exit" input
        while True:
            self.POSITION_IDS, inputs, image_offset, pixel_num = get_position_ids(processor=self.processor,
                                                                       config=self.config,
                                                                       video_path=self.video_path)
            position_ids = self.POSITION_IDS
            
            pixel_values = inputs.pixel_values_videos
            grid_thw = inputs.video_grid_thw
            cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0, dtype=torch.int32
            )
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

            attention_mask_vit = torch.full(
                [1, pixel_values.shape[0], pixel_values.shape[0]], torch.finfo(torch.float32).min, dtype=torch.float32
            )
            cu_seqlens_list = cu_seqlens.flatten().tolist()
            for i in range(1, len(cu_seqlens)):
                attention_mask_vit[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

            pos_ids = []
            for t, h, w in grid_thw:
                hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
                hpos_ids = hpos_ids.reshape(
                    h // self.config['vision_config']['spatial_merge_size'],
                    self.config['vision_config']['spatial_merge_size'],
                    w // self.config['vision_config']['spatial_merge_size'],
                    self.config['vision_config']['spatial_merge_size'],
                )
                hpos_ids = hpos_ids.permute(0, 2, 1, 3)
                hpos_ids = hpos_ids.flatten()

                wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
                wpos_ids = wpos_ids.reshape(
                    h // self.config['vision_config']['spatial_merge_size'],
                    self.config['vision_config']['spatial_merge_size'],
                    w // self.config['vision_config']['spatial_merge_size'],
                    self.config['vision_config']['spatial_merge_size'],
                )
                wpos_ids = wpos_ids.permute(0, 2, 1, 3)
                wpos_ids = wpos_ids.flatten()
                pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
            pos_ids = torch.cat(pos_ids, dim=0)

            print("[DEBUG] pixel_values_videos - num_tokens: {}, feat_dim: {}".format(
                inputs.pixel_values_videos.shape[0],
                inputs.pixel_values_videos.shape[1]
            ))

            print("[DEBUG] pos_ids - shape: {}".format(
                pos_ids.shape
            ))

            # prefill vit
            pixel_values_prefill = torch.zeros([2000, 1176]).to(dtype=torch.float32)
            pixel_values_prefill[:inputs.pixel_values_videos.shape[0],:] = inputs.pixel_values_videos
            pos_ids_prefill = torch.zeros([2000, 2]).to(dtype=torch.int32)
            pos_ids_prefill[:pos_ids.shape[0],:] = pos_ids
            attention_mask_vit_prefill = torch.zeros([1, 2000, 2000], dtype=torch.bool)
            # attention_mask_vit_prefill = torch.ones([1, 2000, 2000], dtype=torch.float) * -10000

            attention_mask_vit_prefill[0,:pos_ids.shape[0],:pos_ids.shape[0]] = attention_mask_vit

            # 对 inputs.input_ids.squeeze(0).tolist() 打印长度和部分信息
            input_ids_list = inputs.input_ids.squeeze(0).tolist()
            print("[DEBUG] input_ids (list) - length: {}, type of element: {}, sample: {}".format(
                len(input_ids_list),
                type(input_ids_list[0]) if len(input_ids_list) > 0 else None,
                input_ids_list[:10]
            ))

            # 对 position_ids.flatten().tolist() 打印长度和部分信息
            position_ids_list = position_ids.flatten().tolist()
            print("[DEBUG] position_ids (list) - length: {}, type of element: {}, sample: {}".format(
                len(position_ids_list),
                type(position_ids_list[0]) if len(position_ids_list) > 0 else None,
                position_ids_list[:10]
            ))

            # 对 pixel_values_prefill.flatten().tolist() 打印长度和部分信息
            pixel_values_list = pixel_values_prefill.flatten().tolist()
            print("[DEBUG] pixel_values (list) - length: {}, type of element: {}, sample: {}".format(
                len(pixel_values_list),
                type(pixel_values_list[0]) if len(pixel_values_list) > 0 else None,
                pixel_values_list[:10]
            ))

            # 对 pos_ids_prefill.flatten().tolist() 打印长度和部分信息
            pos_ids_list = pos_ids_prefill.flatten().tolist()
            print("[DEBUG] pos_ids (list) - length: {}, type of element: {}, sample: {}".format(
                len(pos_ids_list),
                type(pos_ids_list[0]) if len(pos_ids_list) > 0 else None,
                pos_ids_list[:10]
            ))

            print("[DEBUG] cu_seqlens_list - length: {}, type of element: {}, sample: {}".format(
                len(cu_seqlens_list),
                type(cu_seqlens_list[0]) if len(cu_seqlens_list) > 0 else None,
                cu_seqlens_list[:]
            ))

            attnmask = attention_mask_vit_prefill.flatten().tolist()
            print("[DEBUG] attention_mask_vit_list - length: {}, type of element: {}, sample: {}".format(
                len(attnmask),
                type(attnmask[0]) if len(attnmask) > 0 else None,
                attnmask[:10]
            ))

            # 打印 image_offset 和 actual_video_tokens（均为 int）
            print("[DEBUG] image_offset (int): {}".format(image_offset))
            print("[DEBUG] actual_video_tokens (int): {}".format(pixel_num))

            # Chat
            first_start = time.time()
            token = self.model.forward_first(input_ids_list, position_ids_list, pixel_values_list,
                                             pos_ids_list, attnmask,
                                             image_offset, pixel_num)
            first_end = time.time()
            tok_num = 1
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IM_END, self.ID_END
                                ] and self.model.token_length < self.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens,
                                             skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode(
                            [token, token],
                            skip_special_tokens=True)[len(pre_word):]
                    text += word
                    print(word, flush=True, end="")
                    full_word_tokens = []
                token = self.model.forward_next()
                tok_num += 1
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Qwen2VL(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        help='path to the bmodel file')
    parser.add_argument('-t',
                        '--tokenizer_path',
                        type=str,
                        default="../../support/token_config",
                        help='path to the tokenizer file')
    parser.add_argument('-p',
                        '--processor_path',
                        type=str,
                        default="./processor_config",
                        help='path to the processor file')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default="../../compile/files/Qwen2-VL-2B-Instruct/config.json",
                        help='path to the model config file')
    parser.add_argument('-d', '--devid', type=int,
                        default=0, help='device ID to use')
    parser.add_argument('-l', '--visual_length', type=int,
                        default=2000, help='visual length')
    parser.add_argument('-g',
                        '--generation_mode',
                        type=str,
                        choices=["greedy", "penalty_sample"],
                        default="greedy",
                        help='mode for generating next token')
    parser.add_argument('-v',
                        '--video_path',
                        type=str,
                        required=True,
                        default="/data/LLM-TPU/models/Qwen2_VL/python_demo_video/python_demo/test1.mp4",
                        help='path to the video file')
    args = parser.parse_args()
    main(args)