# 根据摘要生成代码prompt生成代码，并将其中```部分替换为原始代码，计算loss作为模型对摘要的置信度
# 利用ms-swift完成模型的加载、template设置等
# 接受的数据类型为要求模型根据摘要生成代码，label部分为参考代码
import swift
from swift.llm.infer.infer import *
from swift.llm import to_device
import torch
import re
from template import *

# 当前仅支持pt后端
class SpmiEval(SwiftInfer):
    def __init__(self, args = None):
        super().__init__(args)
        self.args.logprobs = True
        assert self.args.infer_backend == 'pt', "当前仅支持pt后端"

    def infer_dataset(self) -> List[Dict[str, Any]]:
        args = self.args
        request_config = args.get_request_config()
        logger.info(f'request_config: {request_config}')

        val_dataset = self._prepare_val_dataset()
        logger.info(f'val_dataset: {val_dataset}')

        self.infer_kwargs['metrics'] = [InferStats()]

        if args.write_batch_size <= 0:
            args.write_batch_size = len(val_dataset)
        if args.write_batch_size < len(val_dataset) and args.result_path:
            logger.info(f'args.result_path: {args.result_path}')
        prog_bar = tqdm(
            total=len(val_dataset), dynamic_ncols=True, disable=args.write_batch_size >= len(val_dataset))
        result_list = []
        idx = 0
        while idx < len(val_dataset):
            shard_size = min(args.write_batch_size, len(val_dataset) - idx, args.max_batch_size)
            shard_dataset = val_dataset.select(range(idx, idx + shard_size))
            result_list += self._batch_infer(shard_dataset, request_config)
            idx += shard_size
            prog_bar.update(shard_size)
        prog_bar.close()
        metrics = self.infer_kwargs.pop('metrics')
        if result_list:
            metric = metrics[0].compute()
            print(f'[rank{args.rank}] {metric}' if args.rank >= 0 else str(metric))

        all_spmi = [res['spmi'] for res in result_list]
        avg_spmi = sum(all_spmi) / len(all_spmi) if all_spmi else 0.0
        logger.info(f'Average spmi: {avg_spmi}')
        return result_list

    def _batch_infer(self, val_dataset, request_config):
        args = self.args    
        result_list = []
        rank, data_parallel_size = args.rank, args.global_world_size
        if rank >= 0 and data_parallel_size > 1:
            val_dataset = val_dataset.shard(data_parallel_size, rank, contiguous=True)
        val_dataset = list(val_dataset)
        labels_list = []
        for data in val_dataset:
            labels = data.pop('labels')
            labels_list.append(labels)

        spmis = self._cal_spmi(val_dataset, labels_list)
        if not (args.infer_backend == 'vllm' and rank >= 0 and args.rank % args.vllm_tensor_parallel_size != 0):
            for data, spmi, labels in zip(val_dataset, spmis, labels_list):
                data = {'spmi': spmi, 'labels':labels, **data}
                result_list.append(data)
        if self.jsonl_writer:
            self.jsonl_writer.append(result_list, gather_obj=True)
        return result_list

    def _mask_match(self, encodeds: dict, patterns: str, logits: torch.Tensor) -> torch.BoolTensor:
        # 构建有效token掩码,logits shape (batch_size, seq_len,vocab_size)
        valid_mask = torch.zeros((logits.size(0),logits.size(1)), dtype=torch.bool, device=logits.device)

        for idx,(encoded, pattern) in enumerate(zip(encodeds, patterns)): 
            content = encoded['context_list'][-1]
            offset_mapping = encoded['raw_tokens'][-1].data['offset_mapping']
            offset_base = sum([len(en.data['offset_mapping']) for en in encoded['raw_tokens'][0:-1]])
            start = content.find(pattern)
            end = start + len(pattern)
            
            token_start, token_end = -1, len(offset_mapping)
            for i, (s, e) in enumerate(offset_mapping):
                if s <= start and e > start:
                    token_start = i
                if s <= end and e > end:
                    token_end = i
                    break
            # assert pattern in ''.join([encoded['context_list'][-1][s:e] for s,e in offset_mapping][token_start:token_end+1])
            valid_mask[idx,offset_base+token_start:offset_base+token_end+1] = True  # logits/shift_labels 有效位置
        return valid_mask

    def _cal_spmi(self, val_dataset, labels:List[str]) -> Dict:
        """
            spmi = \frac{\pi(q \mid s,p)}{\pi(q \mid s,p) + \pi(q)}
        """
        with torch.no_grad():
            model = self.infer_engine.model
            device = model.device
            template = self.template
            celoss = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            tokenizer_kwargs = {"return_offsets_mapping": True}

            # 将messages中最后一次回答的代码片段替换为labels中的代码
            messages = [data['messages'] for data in val_dataset]
            for message, label in zip(messages, labels):
                content = message[-1]['content']
                reversed_content = content[::-1]
                pattern = re.compile(rf'```(.*?)```', re.DOTALL)
                matches = list(pattern.finditer(reversed_content))
                start, end = -1,-1
                if matches:
                    last_match = matches[0]
                    start, end = last_match.span()
                    # 只替换最后一个代码块
                    message[-1]['content'] = content[:-end] + f'```\n{label}\n```'
                    if start!=0:
                        message[-1]['content'] += content[-start:]
                else:
                    # 没有找到代码块，直接替换为label
                    message[-1]['content'] = label

            # cal \pi(q|s,p)
            inputs = [StdTemplateInputs.from_dict({'messages':message}) for message in messages]
            inputs = [TemplateInputs(chosen=input) for input in inputs]
            encodeds = [template.encode_with_kwargs(input, **tokenizer_kwargs) for input in inputs]
            batch_encoded_inputs = to_device(template.data_collator(encodeds), device)
            outputs = model(**batch_encoded_inputs)
            logits = outputs.logits
            valid_masks = self._mask_match(encodeds, labels,logits)
            labels = [torch.tensor(encoded['labels'],device=logits.device) for encoded in encodeds]
            labels = pad_sequence(labels, batch_first=True, padding_value=-100).contiguous()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:,1:].contiguous()
            shift_masks = valid_masks[:,:-1].contiguous()
            token_probs = celoss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view_as(shift_labels)
            token_probs = token_probs * shift_masks
            token_probs = torch.exp(-token_probs)  # 转为概率
            # 将有效token移至最左端，辅助先验后验向量对齐
            mask_start = shift_masks.int().argmax(dim=1, keepdim=True)  # 每行第一个True的位置
            mask_len = shift_masks.sum(dim=1)
            token_align = torch.zeros_like(token_probs)
            for i in range(token_probs.size(0)):
                token_align[i,:mask_len[i]] = token_probs[i,mask_start[i,0]:mask_start[i,0]+mask_len[i]]

            valid_len = valid_masks.sum(dim=1)
            
            # cal spmi
            spmi = token_align
            spmi_list = []
            for i in range(spmi.size(0)):
                if valid_len[i]>0:
                    spmi_list.append(spmi[i,:valid_len[i]].nanmean().item())
                else:
                    spmi_list.append(0.0)
        return spmi_list
    
def main(args: Optional[Union[List[str], InferArguments]] = None):
    return SpmiEval(args).main()

if __name__ == "__main__":
    main()
