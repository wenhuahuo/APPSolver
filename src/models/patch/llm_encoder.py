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
        
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModel.from_pretrained(model_name)
        
        for param in self.llm.parameters():
            param.requires_grad = False

        if model_name == 'Qwen/Qwen2.5-0.5B':
            self.hidden_size = self.llm.config.hidden_size
        elif model_name == 'Qwen/Qwen3.5-0.8B':
            self.hidden_size = self.llm.config.text_config.hidden_size
        
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
            outputs = self.llm(**inputs)
            
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                cls_embed = outputs.last_hidden_state[:, 0, :]
            else:
                cls_embed = outputs.pooler_output
            
            cls_embed = cls_embed.float()
        
        return cls_embed.squeeze(0)
