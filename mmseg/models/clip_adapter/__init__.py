import logging
import torch
from .text_prompt import (
    PredefinedPromptExtractor,
    ImageNetPromptExtractor,
    VILDPromptExtractor,
    LearnablePromptExtractor,
)
from .adapter import ClipAdapter


def build_prompt_learner(
    prompt_type,
    prompt_dim=None,
    prompt_shape=None,
    prompt_templates=["{}"],
    checkpoint_path="",
):
    if prompt_type == "predefined":
        prompt_learner = PredefinedPromptExtractor(prompt_templates)
    elif prompt_type == "imagenet":
        prompt_learner = ImageNetPromptExtractor()
    elif prompt_type == "vild":
        prompt_learner = VILDPromptExtractor()
    elif prompt_type == "learnable":
        prompt_learner = LearnablePromptExtractor(
            prompt_dim=prompt_dim,
            prompt_shape=prompt_shape,
        )
        if checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
            missing, unexpected = prompt_learner.load_state_dict(
                {
                    ".".join(k.split(".")[2:]): v
                    for k, v in checkpoint.items()
                    if "prompt_learner" in k
                },
                strict=False,
            )
            for param in prompt_learner.parameters():
                param.requires_grad = False
            prompt_learner.with_trainable_params = False
            print("Load Prompt Learner from {}".format(checkpoint_path))
            print("Missing {}".format(missing))
            print("Unexpected {}".format(unexpected))
        else:
            trainable_params = [
                k
                for k, v in prompt_learner.named_parameters()
                if v.requires_grad == True
            ]
            print("Prompt Learner training params: {}".format(trainable_params))
    else:
        raise NotImplementedError(
            "Prompt learner {} is not supported".format(prompt_type)
        )
    return prompt_learner
