import inspect
from functools import partial
import torch
from typing import Any, Dict, Optional
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    DeepLift,
    GuidedBackprop,
    GuidedGradCam,
    Deconvolution,
    LRP,
    GradientShap,
    ShapleyValueSampling
)
from torch.nn import functional as F
import transformers


ATTR_NAME_ALIASES = {
    'ig': 'integrated_gradients',
    'saliency': 'gradient',
    'dl': 'deep_lift',
    'gb': 'guided_backprop',
    'gg': 'guided_gradcam',
    'deconv': 'deconvolution',
    'lrp': 'layer_relevance_propagation',
    'gs': 'gradient_shap',
    'grad_shap': 'gradient_shap',
    'shap': 'shapley_values'
}

ATTR_NAME_TO_CLASS = { # TODO: Add more Captum Primary attributions with needed computed arguments
    'integrated_gradients': IntegratedGradients,
    'gradient': Saliency,
    'grad_x_input': InputXGradient,
    'deep_lift': DeepLift,
    'guided_backprop': GuidedBackprop,
    'guided_gradcam': GuidedGradCam,
    'deconvolution': Deconvolution,
    'layer_relevance_propagation': LRP,
    'gradient_shap': GradientShap,
    'shapley_values': ShapleyValueSampling
}


def compute_primary_attributions_scores(attr_method : str, model: transformers.PreTrainedModel,
                                        forward_kwargs: Dict[str, Any], prediction_id: torch.Tensor,
                                        aggregation: str = "L2", normalize: bool = True,
                                        attribution_kwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    """
    Computes the primary attributions with respect to the specified `prediction_id`.

    Args:
        attr_method: Name of the primary attribution method to compute
        model: HuggingFace Transformers Pytorch language model.
        forward_kwargs: contains all the inputs that are passed to `model` in the forward pass
        prediction_id: Target Id. The Integrated Gradients will be computed with respect to it.
        aggregation: Aggregation method to perform to the Integrated Gradients attributions.
         Currently only "L2" is implemented
        normalize: whether to normalize or not the obtained attributions
        attribution_kwargs: extra arguments used to pass to Captum's `attribute` method

    Returns: a tensor of the normalized attributions with shape (input sequence size,)
    """

    def model_forward(input_: torch.Tensor, decoder_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
            -> torch.Tensor:
        if decoder_ is not None:
            output = model(inputs_embeds=input_, decoder_inputs_embeds=decoder_, **extra_forward_args)
        else:
            output = model(inputs_embeds=input_, **extra_forward_args)
        return F.softmax(output.logits[:, -1, :], dim=-1)

    def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
        # attributes has shape (batch, sequence size, embedding dim)
        attributes = attributes.squeeze(0)

        if aggregation == "L2":
            attributes = torch.norm(attributes, dim=1) # norm calculates a scalar value (L2 Norm)
        else:
            raise NotImplemented

        if normalize:
            attributes = attributes / torch.sum(attributes)  # Normalize the values so they add up to 1

        return attributes.detach().cpu()

    extra_forward_args = {k: v for k, v in forward_kwargs.items() if
                          k not in ['inputs_embeds', 'decoder_inputs_embeds']}
    input_ = forward_kwargs.get('inputs_embeds')
    decoder_ = forward_kwargs.get('decoder_inputs_embeds')

    if decoder_ is None:
        forward_func = partial(model_forward, decoder_=decoder_, model=model, extra_forward_args=extra_forward_args)
        inputs = input_
        baselines = torch.randn(20, *tuple(inputs.shape[1:]), device=inputs.device)
        feature_mask = torch.arange(0, inputs.shape[0], device=inputs.device).repeat(1, inputs.shape[2], 1).transpose(1, 2)

    else:
        forward_func = partial(model_forward, model=model, extra_forward_args=extra_forward_args)
        inputs = tuple([input_, decoder_])
        baselines = (
            torch.randn(20, *tuple(input_.shape)[1:], device=input_.device),
            torch.randn(20, *tuple(decoder_.shape[1:]), device=decoder_.device)
        )
        feature_mask = tuple(
            torch.arange(0, inp.shape[0], device=inp.device).repeat(1, inp.shape[2], 1).transpose(1, 2)
            for inp in [input_, decoder_]
        )

    # Get attributions method class
    attr_method_class = ATTR_NAME_TO_CLASS.get(ATTR_NAME_ALIASES.get(attr_method, attr_method), None)
    if attr_method_class is None:
        raise NotImplementedError(
            f"No implementation found for primary attribution method '{attr_method}'. "
            f"Please choose one of the methods: {list(ATTR_NAME_TO_CLASS.keys())}"
        )

    # Get attributions
    attr_obj = attr_method_class(forward_func=forward_func)
    attribute_args = {'inputs': inputs, 'target': prediction_id, **(attribution_kwargs or {})}
    if 'baselines' in inspect.signature(attr_obj.attribute).parameters:
        attribute_args['baselines'] = baselines
    if 'feature_mask' in inspect.signature(attr_obj.attribute).parameters:
        attribute_args['feature_mask'] = feature_mask
    attributions = attr_obj.attribute(**attribute_args)

    # Return attributions
    if isinstance(attributions, tuple):
        # Does it make sense to concatenate encoder and decoder attributions before normalization?
        # We assume that the encoder/decoder embeddings are the same
        norm_attributions =  normalize_attributes(torch.cat(attributions, dim=1))
    else:
        norm_attributions = normalize_attributes(attributions)

    # clear cuda cache and return
    torch.cuda.empty_cache()
    return norm_attributions