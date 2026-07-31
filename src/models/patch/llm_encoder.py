"""
LLM-based Encoder for Ship Parameters

Uses a frozen LLM (e.g., Qwen2.5-0.5B) to extract raw hidden features
from ship parameter text. No task projection is applied at pre-compute time.
"""

import torch
import torch.nn as nn


class LLMEncoder(nn.Module):
    """
    Frozen LLM encoder that extracts raw hidden features from ship parameters text.

    Args:
        model_name: HuggingFace model name for the LLM
    """
    
    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2.5-0.5B',
    ):
        super().__init__()
        self.model_name = model_name
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        
        for param in self.llm.parameters():
            param.requires_grad = False

        config = self.llm.config
        if hasattr(config, 'hidden_size'):
            self.hidden_size = config.hidden_size
        else:
            self.hidden_size = config.text_config.hidden_size
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(
        self,
        text: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            text: Ship parameters text string
            device: Target device
             
        Returns:
            (hidden_size,) raw LLM feature vector
        """
        self.llm.to(device)
        
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors='pt',
        ).to(device)
        
        with torch.no_grad():
            outputs = self.llm(
                **inputs, output_hidden_states=True, use_cache=False
            )
            hidden = outputs.hidden_states[-1]
            last_index = inputs['attention_mask'].sum(dim=1) - 1
            batch_index = torch.arange(last_index.shape[0], device=device)
            cls_embed = hidden[batch_index, last_index].float()
        
        return cls_embed.squeeze(0)
