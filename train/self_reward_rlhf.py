# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Union

from swift.llm import safe_snapshot_download
from swift.utils import get_logger, get_model_parameter_info
from swift.llm.argument import BaseArguments, RLHFArguments
from swift.llm.model import HfConfigFactory
from swift.llm.train.rlhf import SwiftRLHF

from .trainer import InerRewardTrainer
# from .outer_trainer import OuterRewardTrainer
from .template import *

logger = get_logger()


class SELF_REWARD_RLHF(SwiftRLHF):
    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()

        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        args.save_args()

        data_collator = self._get_data_collator()
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')

        trainer = InerRewardTrainer(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)

def self_reward_rlhf_main(args: Union[List[str], RLHFArguments, None] = None):
    # 目前仅支持GRPO类型的self-reward RLHF
    return SELF_REWARD_RLHF(args).main()
