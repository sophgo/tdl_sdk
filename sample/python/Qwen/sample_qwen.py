#!/usr/bin/env python3
import sys
import time
import signal
import argparse
from transformers import AutoTokenizer
from tdl import llm

class QwenChat:
    def __init__(self, args):
        # 初始化tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )
        
        # 预热tokenizer
        self.tokenizer.decode([0])
        
        # 初始化聊天历史和系统提示词
        self.system_prompt = "You are a helpful assistant."
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.EOS = self.tokenizer.eos_token_id
        self.enable_history = args.enable_history
        
        # 初始化模型
        print("\n正在加载模型:", args.model_path)
        self.model = llm.Qwen()

        self.model.model_open(args.model_path)


    def clear(self):
        """清除历史记录"""
        self.history = [{"role": "system", "content": self.system_prompt}]
        
    def encode_tokens(self, input_str):
        """将输入文本编码为token"""
        # 使用正确的chat template
        self.history.append({"role": "user", "content": input_str})
        text = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(text)
        

    def update_history(self, answer):
        """更新历史"""
        if not self.enable_history:
            self.clear()
            return
            
        self.history.append({"role": "assistant", "content": answer})
        
    def chat(self):
        """开始聊天会话"""
        print(
            """\n=================================================================
1. 如果要退出，请输入: q, quit, exit
2. 如果要开始新的对话，请输入: clear, new
================================================================="""
        )
        
  
        while True:
            # 获取用户输入
            input_str = input("\n问题: ")
            
            # 处理特殊命令
            if input_str.lower() in ["exit", "q", "quit"]:
                break
            elif input_str.lower() in ["clear", "new"]:
                self.clear()
                continue
            elif not input_str.strip():
                continue
                
            # 编码输入
            tokens = self.encode_tokens(input_str)
            if not tokens:
                print("输入为空，请重新输入")
                continue
                
            print("\n回答: ", end="", flush=True)
            

            self.stream_answer(tokens)

    def stream_answer(self, tokens):
        """带预定义回答的流式生成"""
        # 预定义回答
        predefined_answers = {

        }
        
        # 检查是否有预定义回答
        input_text = self.tokenizer.decode(tokens)
        for key, answer in predefined_answers.items():
            if key in input_text:
                print(answer)
                return
        
        # 如果没有匹配的预定义回答，尝试模型生成
        try:
            tok_num = 0
            answer_tokens = []
            full_word_tokens = []
            
            # 生成第一个token
            first_start = time.time()
            token = self.model.inference_first(tokens)
            first_end = time.time()
            
            # 跟踪重复token
            consecutive_same = 0
            last_token = None
            
            # 生成后续token
            while token != self.EOS and tok_num < self.max_generation_tokens:
                # 检测重复
                if token == last_token:
                    consecutive_same += 1
                    if consecutive_same >= self.max_consecutive_same:
                        print(f"\n[检测到重复生成: {consecutive_same}次相同token {token}，已中断]")
                        break
                else:
                    consecutive_same = 0
                    
                last_token = token
                
                # 添加到待解码列表
                full_word_tokens.append(token)
                
                # 尝试解码
                word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                if "�" in word:  # 不完整的unicode字符
                    token = self.model.inference_next()
                    tok_num += 1
                    continue
                    
                # 解码成功，输出并重置
                print(word, flush=True, end="")
                answer_tokens.extend(full_word_tokens)
                full_word_tokens = []
                tok_num += 1
                
                # 生成下一个token
                token = self.model.inference_next()
                
            # 计算性能指标
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration if next_duration > 0 else 0
            
            # 更新历史
            answer = self.tokenizer.decode(answer_tokens)
            self.update_history(answer)
            
            print()
            print(f"FTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")
        except Exception as e:
            print(f"\n生成失败: {e}")
            print("很抱歉，我暂时无法回答这个问题。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="./token_config", help='tokenizer配置路径')
    parser.add_argument('--enable_history', action='store_true', help="启用历史记忆功能")
    args = parser.parse_args()
    
    chat = QwenChat(args)
    chat.chat()

if __name__ == "__main__":
    main()