import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional,Any, Callable, Tuple, Union

import json
import torch
import re

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import ORM, orms

"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""
import textstat
import re
from typing import List

class FleschReadingEase(ORM):
    """
    Reward function that calculates the Flesch Reading Ease score 
    and normalizes it to the 0-1 range for each completion.
    """
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            raw_score = textstat.flesch_reading_ease(content)
            # 归一化到0-1范围
            raw_score = max(0.0, min(raw_score, 100.0))
            reward = raw_score / 100.0
            rewards.append(reward)
            
        return rewards

class ChinesePrecent(ORM):
    """
    通过中文占比给予奖励函数
    """
    def __call__(self, completions, **kwargs):
        rewards = []
        for content in completions:
            # 计算中文字符占比
            chinese_chars = re.findall(r'[\u4e00-\u9fa5]', content)
            total_chars = len(content)
            if total_chars > 0:
                chinese_ratio = len(chinese_chars) / total_chars
            else:
                chinese_ratio = 0.0
            rewards.append(chinese_ratio)
        return rewards


InputsType = List[Dict[str, Union[torch.Tensor, Any]]]
class ReverseArithmeticMeanReward(ORM):
    def __init__(self,language='java'):
        super().__init__()
        self.reverse_system = f"Act as a {language} code generation assistant. Reconstruct the original code based on the provided functional summary. Wrap ```{language}\n``` around the code."

    def _set_inputs_reverse_system(self, inputs):
        for _input in inputs:
            messages = _input['messages']
            if messages[0]['role'] == 'system':
                messages[0] = {'role': 'system', 'content': self.reverse_system}
            else:
                messages.insert(0, {'role': 'system', 'content': self.reverse_system})

    def _generate_reverse_completions(self, **kwargs) -> InputsType:
        """Use trainer._generate_completions to generate reverse completions

        """
        inputs = kwargs.get('inputs')
        trainer = kwargs.get('trainer')
        # 修改inputs,使用反向system prompt,将user content替换为assistant content,删除assistant,将user content提取为标签
        inputs = deepcopy(inputs)
        origin_codes = []
        self._set_inputs_reverse_system(inputs)
        for _input in inputs:
            messages = _input['messages']
            assert messages[-1]['role'] == 'assistant'
            assistant_content = messages[-1]['content']
            # filter <think>.*</think> content
            assistant_content = re.sub(r'<think>.*?</think>', '', assistant_content, flags=re.DOTALL)
            messages.pop()  # remove the last assistant message
            origin_codes.append(messages[-1]['content']) # save the user content as label
            messages[-1]['content'] = assistant_content # replace the last user message
        
        # 确保inputs长度不超过self.max_length
        truncation_strategy = trainer.template.truncation_strategy
        trainer.template.truncation_strategy = "right"
        inputs = trainer._generate_completions(inputs)
        trainer.template.truncation_strategy = truncation_strategy
        
        # 替换生成的代码为origin_code,并记录代码位置
        for _input, origin_code in zip(inputs, origin_codes):
            content = _input['messages'][-1]['content']
            reversed_content = content[::-1]
            pattern = re.compile(rf'```(.*?)```', re.DOTALL)
            matches = list(pattern.finditer(reversed_content))
            start, end = -1,-1
            if matches:
                last_match = matches[0]
                start, end = last_match.span()
                # 只替换最后一个代码块
                _input['messages'][-1]['content'] = content[:-end] + f'```cpp\n{origin_code}\n```'
                if start!=0:
                    _input['messages'][-1]['content'] += content[-start:]
            else:
                # 没有找到代码块，对应reward为0
                pass
            # 记录代码位置，方便后续计算loss
            _input['code_text'] = f'```cpp\n{origin_code}\n```' # 记录代码位置，方便后续计算loss
                
        return inputs

    def _mask_match(self, batch_reverse, labels, reverse) -> torch.BoolTensor:
        # 构建有效token掩码
        batch_len = [len(batch['input_ids']) for batch in batch_reverse] # 每个样本的总token长度
        valid_mask = torch.zeros((labels.size(0), labels.size(1) - 1), dtype=torch.bool, device=labels.device)
            
        # 第一步：构建有效token掩码
        for i, (encode_reverse, str_reverse) in enumerate(zip(batch_reverse, reverse)):
            context = encode_reverse['context_list'][-1]
            if isinstance(context,list):    # Template has suffix
                context = encode_reverse['context_list'][-2]
            target_context = str_reverse['code_text']
            offset_mapping = encode_reverse['raw_tokens'][-1].data['offset_mapping']
            offset_base = sum([len(en.data['offset_mapping']) for en in encode_reverse['raw_tokens'][0:-1]])
            start = context.find(target_context)
            end = start + len(target_context)
            
            if start == -1:
                # 标记为无效样本
                continue
            
            token_start, token_end = -1, -1
            for idx, (s, e) in enumerate(offset_mapping):
                if s <= start and e > start:
                    token_start = idx
                if s < end and e >= end:
                    token_end = idx
                    break
            
            if token_start == -1 or token_end == -1:
                continue
            
            # 标记有效token范围
            base_idx = sum(batch_len[:i])
            valid_mask[0, base_idx+offset_base+token_start:base_idx+offset_base+token_end] = True  # logits/shift_labels 有效位置
        return valid_mask

    def __call__(self, completions, **kwargs):
        trainer = kwargs.get('trainer')
        template = trainer.template
        reverse = self._generate_reverse_completions(**kwargs)
        
        with torch.no_grad():
            with trainer._template_context(template):
                truncation_strategy = trainer.template.truncation_strategy
                trainer.template.truncation_strategy = "right"
                
                tokenizer_kwargs = {"return_offsets_mapping": True}
                batch_reverse = [template.encode_with_kwargs(infer_request, return_length=True, **tokenizer_kwargs) for infer_request in reverse]
                batch_len = [len(batch['input_ids']) for batch in batch_reverse] # 每个样本的总token长度
                batch_encoded_inputs = to_device(template.data_collator(batch_reverse.copy()), trainer.model.device)
                
                trainer.template.truncation_strategy = truncation_strategy
            
            outputs = trainer.model(**batch_encoded_inputs) # 所有样本token的拼接
        
            logits = outputs.logits
            labels = batch_encoded_inputs['labels']
            
            valid_mask = self._mask_match(batch_reverse, labels, reverse)
                        
            reward = self._loss_fn(logits, labels, valid_mask, batch_len)
            # 清理GPU中的多余向量
            del batch_encoded_inputs,logits, labels, valid_mask, outputs
            torch.cuda.empty_cache()
            return reward
    
    def _loss_fn(self, logits, labels, valid_mask, batch_len):
        """计算有效token的平均loss
            sample_loss = mean(P)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).view_as(shift_labels)
        
        token_reward = torch.exp(token_losses * -1)
        masked_reward = token_reward * valid_mask.float()
        avg_reward = []
        for i in range(len(batch_len)):
            valid_token_count = valid_mask[0, sum(batch_len[:i]):sum(batch_len[:i+1])].sum()
            if valid_token_count > 0:
                avg_reward.append(masked_reward[0, sum(batch_len[:i]):sum(batch_len[:i+1])].sum() / valid_token_count)
            else:
                avg_reward.append(0.0)
        return avg_reward
        
class ReverseGeometricMeanReward(ReverseArithmeticMeanReward):
    def _loss_fn(self, logits, labels, valid_mask, batch_len):
        """计算有效token的平均loss
            sample_loss = exp(-mean(celoss))
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        ).view_as(shift_labels)
        
        masked_reward = token_losses * valid_mask.float()
        geo_mean_reward = []
        for i in range(len(batch_len)):
            valid_token_count = valid_mask[0, sum(batch_len[:i]):sum(batch_len[:i+1])].sum()
            if valid_token_count > 0:
                mean_loss = masked_reward[0, sum(batch_len[:i]):sum(batch_len[:i+1])].sum() / valid_token_count
                geo_mean_reward.append(torch.exp(-mean_loss))
            else:
                geo_mean_reward.append(0.0)
        return geo_mean_reward


orms['external_flesch_reading_ease'] = FleschReadingEase
orms['external_chinese_readability'] = ChinesePrecent
orms['external_reverse_arithmetic'] = ReverseArithmeticMeanReward
orms['external_reverse_geometric'] = ReverseGeometricMeanReward