from swift.trainers.rlhf_trainer.grpo_trainer import *
import torch.nn.functional as F
system_prompt = "You are a helpful assistant that generates C++ code summaries. Given a piece of C++ code, provide a concise summary."
reverse_system_prompt = "Act as a C++ code generation assistant. Reconstruct the original code based on the provided functional summary. Wrap ```cpp\n``` around the code."

# 将对偶学习视为reward的一环
class InerRewardTrainer(GRPOTrainer):
    def _compute_rewards_per_func(self, inputs: DataType) -> torch.Tensor:
        """Compute rewards using all reward functions"""
        device = self.accelerator.device
        rewards_per_func = torch.zeros((len(inputs), len(self.reward_funcs)), device=device)
        completions = [inp['messages'][-1]['content'] for inp in inputs]
        for i, (reward_func, reward_model_plugin, reward_func_name) in enumerate(
                zip(self.reward_funcs, self.reward_model_plugins, self.reward_func_names)):
            with patch_profiling_context(self, reward_func_name):
                # reward model
                reward_kwargs = {'trainer_state': self.state}
                if self.enable_server_multi_turn:
                    trajectory_inputs = self._get_trajectory_inputs(inputs)
                    reward_kwargs.update({'trajectory_inputs': trajectory_inputs})
                if isinstance(reward_func, nn.Module):
                    output_reward_func = reward_model_plugin(inputs=inputs, **reward_kwargs)
                # reward function
                else:
                    # Repeat all input columns (but "messages" and "completion") to match the number of generations
                    reward_kwargs.update(RowPreprocessor.rows_to_batched(inputs))
                    reward_kwargs['trainer'] = self
                    reward_kwargs['inputs'] = inputs
                    output_reward_func = reward_func(completions, **reward_kwargs)
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs['completion'] = completions[nan_row_idx]
            logger.warning(f'All reward functions returned None for the following kwargs: {row_reward_kwargs}. '
                           'Please ensure that at least one reward function returns a valid reward.')

        return rewards_per_func
               