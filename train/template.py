# 替换qwen3的template以支持tokenizer_kwargs参数传递
from swift.llm.template.template.qwen import Qwen3Template
from swift.llm.template.base import *
from swift.llm.template.register import register_template,TemplateMeta


class TokenizerKwargsTemplate(Template):
    @torch.inference_mode()
    def encode_with_kwargs(self,
               inputs: Union[TemplateInputs, Dict[str, Any], InferRequest],
               return_template_inputs: bool = False,
               return_length: bool = False,
               **tokenizer_kwargs) -> Dict[str, Any]:
        """The entrance method of Template!

        Returns:
            return {'input_ids': List[int], 'labels': Optional[List[int]], ...}
        """
        assert self._processor_inited, ('Please initialize the processor before calling the template.encode method: '
                                        'template.init_processor(processor).')
        if isinstance(inputs, InferRequest):
            inputs = asdict(inputs)

        if isinstance(inputs, dict):
            if self.task_type == 'causal_lm' and not self.is_training:
                InferRequest.remove_response(inputs['messages'])
            inputs = TemplateInputs.from_dict(inputs)
        elif isinstance(inputs, TemplateInputs):
            inputs = deepcopy(inputs)
        assert isinstance(inputs, TemplateInputs)

        chosen = inputs.chosen
        if self.task_type == 'causal_lm':
            if self.mode in {'train', 'pt', 'vllm', 'lmdeploy', 'sglang'}:
                encoded = self._encode_truncated_with_kwargs(chosen,**tokenizer_kwargs)
            elif self.mode == 'rlhf':
                encoded = self._rlhf_encode(inputs)
            elif self.mode == 'kto':
                encoded = self._kto_encode(inputs)
            elif self.mode == 'gkd':
                encoded = self._gkd_encode(chosen)
        elif self.task_type == 'seq_cls':
            if self.mode == 'rlhf':
                encoded = self._rlhf_encode(inputs)
                for prefix in ['chosen', 'rejected']:
                    encoded.pop(f'{prefix}_labels', None)
                    encoded.pop(f'{prefix}_loss_scale', None)
            else:
                encoded = self._seq_cls_encode(chosen)
        elif self.task_type == 'prm':
            encoded = self._encode_truncated(chosen)
        elif self.task_type == 'embedding':
            encoded = self._embedding_encode(inputs)
        elif self.task_type in {'reranker', 'generative_reranker'}:
            encoded = self._reranker_encode(inputs)
        else:
            raise ValueError(f'task_type: {self.task_type} is not supported.')

        if chosen.channel is not None:
            encoded['channel'] = chosen.channel

        lengths = [0]
        for key in list(encoded.keys()):
            if encoded[key] is None:
                encoded.pop(key)
            elif key.endswith('length'):
                value = encoded[key]
                if isinstance(value, int):
                    lengths.append(value)
                elif isinstance(value, (tuple, list)):
                    lengths += value
        if return_length:
            encoded['length'] = sum(lengths)
        else:
            encoded.pop('length', None)
        if return_template_inputs:
            encoded['template_inputs'] = chosen
        if not self.remove_unused_columns:
            encoded['_extra_kwargs'] = chosen.extra_kwargs
        return encoded
    
    def _encode_truncated_with_kwargs(self, inputs: StdTemplateInputs,**tokenizer_kwargs):
        self._preprocess_inputs(inputs)
        if self.mode in {'vllm', 'lmdeploy', 'sglang'}:
            encoded = Template._encode(self, inputs)
            keys = ['images', 'audios', 'videos']
            if self.mode == 'vllm':
                keys.append('mm_processor_kwargs')
            for key in keys:
                value = getattr(inputs, key)
                if value:
                    encoded[key] = value
        else:
            encoded = self._encode_with_kwargs(inputs,**tokenizer_kwargs)
        self._handle_megatron_cp(encoded)  # TODO: fix cp_size & cached_dataset
        input_ids = encoded.get('input_ids')
        labels = encoded.get('labels')
        loss_scale = encoded.get('loss_scale')
        length = self._get_length(input_ids, labels)
        if self.max_length is not None and length > self.max_length:
            if self.truncation_strategy in {'right', 'left'}:
                input_ids, labels, loss_scale = self._truncate(
                    input_ids, labels, loss_scale, truncation_strategy=self.truncation_strategy)
                length = self._get_length(input_ids, labels)
            elif self.truncation_strategy == 'raise':
                raise MaxLengthError(f'Current length of row({length}) is larger'
                                     f' than the max_length({self.max_length}).')
        encoded['length'] = length
        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        return encoded

    def _encode_with_kwargs(self, inputs: StdTemplateInputs,**tokenizer_kwargs) -> Dict[str, Any]:
        inputs.messages = deepcopy(inputs.messages)
        template_backend = self.template_backend
        if (self.template_meta.template_type == 'dummy' and self.use_chat_template and not self.is_training
                and self.task_type != 'seq_cls'):
            template_backend = 'jinja'
            logger.info_once(f'Setting template_backend: {template_backend}')
        res_context_list, loss_scale_list, answer_len = (
            self._swift_encode(inputs) if template_backend == 'swift' else self._jinja_encode(inputs))
        encoded = {}
        if self.is_encoder_decoder or self.mode == 'gkd':
            total_len = len(res_context_list)
            for key, _slice in zip(['prompt', 'answer'],
                                   [slice(0, total_len - answer_len),
                                    slice(total_len - answer_len, total_len)]):
                context_list, loss_scale = self._simplify_context_list(res_context_list[_slice],
                                                                       loss_scale_list[_slice], inputs)
                input_ids, labels, loss_scale,raw_tokens = self._encode_context_list_with_kwargs(context_list, loss_scale,**tokenizer_kwargs)
                encoded[f'{key}_input_ids'] = input_ids
                encoded[f'{key}_labels'] = labels
                encoded[f'{key}_loss_scale'] = loss_scale
            input_ids = encoded['prompt_input_ids'] + encoded['answer_input_ids']
            labels = encoded['prompt_labels'] + encoded['answer_labels']
            loss_scale = None
            if isinstance(encoded['prompt_loss_scale'], list):
                loss_scale = encoded['prompt_loss_scale'] + encoded['answer_loss_scale']
        else:
            res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, inputs)
            input_ids, labels, loss_scale,raw_tokens = self._encode_context_list_with_kwargs(
                res_context_list, loss_scale_list,**tokenizer_kwargs)
        self._add_dynamic_eos(input_ids, labels, loss_scale, self._encode_context_list_with_kwargs(self.template_meta.suffix,**tokenizer_kwargs)[0])

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        if encoded.get('labels') is not None:
            encoded['labels'][0] = -100
        if encoded.get('loss_scale') is not None:
            encoded['loss_scale'][0] = 0
        if raw_tokens:
            encoded['context_list'] = res_context_list
            encoded['raw_tokens'] = raw_tokens
        return encoded
    
    def _encode_context_list_with_kwargs(
            self,
            context_list: List[Context],
            loss_scale_list: Optional[List[float]] = None,
            **tokenizer_kwargs) -> Tuple[List[int], List[int], List[float], Dict[str, Any]]:
        """return: input_ids, labels, tokenizer_kwargs"""
        input_ids: List[int] = []
        raw_tokens = []
        labels: List[int] = []
        loss_scale: List[float] = []
        if loss_scale_list is None:
            loss_scale_list = [0.] * len(context_list)
        for i, (context, loss_weight) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str):
                raw_token = self._tokenize_with_kwargs(context,**tokenizer_kwargs)
                token_list = raw_token['input_ids']
                raw_tokens.append(raw_token)
            else:
                token_list = context
            input_ids += token_list
            if loss_scale_list[i] > 0.0:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
            if not self.loss_scale.is_binary:
                loss_scale.extend([loss_weight] * len(token_list))
        if self.loss_scale.is_binary:
            loss_scale = None
        return input_ids, labels, loss_scale, raw_tokens

    def _tokenize_with_kwargs(self, text: str, **tokenizer_kwargs) -> List[int]:
        """Tokenize text with additional tokenizer kwargs."""
        return self.tokenizer(
            text, return_attention_mask=False, add_special_tokens=False, **tokenizer_kwargs)


class TokenizerKwargsThinkingTemplate(Qwen3Template,TokenizerKwargsTemplate):
    pass


from swift.llm.template.template.qwen import QwenTemplateMeta, LLMTemplateType
register_template(QwenTemplateMeta(LLMTemplateType.qwen3, default_system=None, template_cls=TokenizerKwargsThinkingTemplate), exist_ok=True)

from swift.llm.template.template.llama import Llama3_2TemplateMeta
register_template(Llama3_2TemplateMeta(LLMTemplateType.llama3_2,template_cls=TokenizerKwargsTemplate),exist_ok=True)

from swift.llm.template.template.glm import GLM4Template, GLM4TemplateMeta
class TokenizerKwargsGLM4Template(GLM4Template,TokenizerKwargsTemplate):
    pass
register_template(GLM4TemplateMeta(LLMTemplateType.glm4, template_cls=TokenizerKwargsGLM4Template),exist_ok=True)


from swift.llm.template.template.deepseek import DeepseekTemplateMeta
register_template(DeepseekTemplateMeta(LLMTemplateType.deepseek, template_cls=TokenizerKwargsTemplate), exist_ok=True)

from swift.llm.template.template.gemma import GemmaTemplateMeta
register_template(GemmaTemplateMeta(LLMTemplateType.gemma, template_cls=TokenizerKwargsTemplate), exist_ok=True)

from swift.llm.template.template.microsoft import Phi4TemplateMeta,Phi4TemplateMeta
register_template(Phi4TemplateMeta(LLMTemplateType.phi4, template_cls=TokenizerKwargsTemplate), exist_ok=True)


from swift.llm.template.template.llm import ThinkingWithAnswerTemplate
class HunyuanThinkingTemplate(ThinkingWithAnswerTemplate,TokenizerKwargsTemplate):
    pass

register_template(
    TemplateMeta(
        LLMTemplateType.hunyuan,
        prefix=['<｜hy_begin▁of▁sentence｜>'],
        system_prefix=['<｜hy_begin▁of▁sentence｜>{{SYSTEM}}<｜hy_place▁holder▁no▁3｜>'],
        prompt=['<｜hy_User｜>{{QUERY}}<｜hy_Assistant｜>'],
        chat_sep=['<｜hy_place▁holder▁no▁2｜>'],
        suffix=['<｜hy_place▁holder▁no▁2｜>'],
        template_cls=HunyuanThinkingTemplate,
        agent_template='hunyuan_hermes'),
        exist_ok=True)