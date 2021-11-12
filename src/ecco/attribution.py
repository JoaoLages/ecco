import re
from functools import partial
import torch
from typing import Any, Dict, Optional, Tuple, List
from captum.attr import IntegratedGradients
from torch.nn import functional as F
from itertools import combinations
from lime.lime_text import LimeTextExplainer
import numpy as np


def saliency(prediction_logit, encoder_token_ids_tensor_one_hot, decoder_token_ids_tensor_one_hot: Optional = None,
             norm=True, retain_graph=False) -> torch.Tensor:

    # only works in batches of 1
    assert len(encoder_token_ids_tensor_one_hot.shape) == 3 and encoder_token_ids_tensor_one_hot.shape[0] == 1
    if decoder_token_ids_tensor_one_hot is not None:
        assert len(decoder_token_ids_tensor_one_hot.shape) == 3 and decoder_token_ids_tensor_one_hot.shape[0] == 1

    # Back-propegate the gradient from the selected output-logit
    prediction_logit.backward(retain_graph=retain_graph)

    token_ids_tensor_one_hot_grad = torch.cat(
        [encoder_token_ids_tensor_one_hot.grad, decoder_token_ids_tensor_one_hot.grad], dim=1
    )[0] if decoder_token_ids_tensor_one_hot is not None else encoder_token_ids_tensor_one_hot.grad[0]

    # token_ids_tensor_one_hot.grad is the gradient propegated to ever embedding dimension of
    # the input tokens.
    if norm:  # norm calculates a scalar value (L2 Norm)
        token_importance_raw = torch.norm(token_ids_tensor_one_hot_grad, dim=1)
        # print('token_importance_raw', token_ids_tensor_one_hot.grad.shape,
        # np.count_nonzero(token_ids_tensor_one_hot.detach().numpy(), axis=1))

        # Normalize the values so they add up to 1
        token_importance = token_importance_raw / torch.sum(token_importance_raw)
    else:
        token_importance = torch.sum(token_ids_tensor_one_hot_grad, dim=1)  # Only one value, all others are zero

    encoder_token_ids_tensor_one_hot.grad.data.zero_()
    if decoder_token_ids_tensor_one_hot is not None:
        decoder_token_ids_tensor_one_hot.grad.data.zero_()

    return token_importance


def saliency_on_d_embeddings(prediction_logit, inputs_embeds, aggregation="L2", retain_graph=True) -> torch.Tensor:
    inputs_embeds.retain_grad()

    # Back-propegate the gradient from the selected output-logit
    prediction_logit.backward(retain_graph=retain_graph)

    # inputs_embeds.grad
    # token_ids_tensor_one_hot.grad is the gradient propegated to ever embedding dimension of
    # the input tokens.
    if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
        token_importance_raw = torch.norm(inputs_embeds.grad, dim=1)
        # print('token_importance_raw', token_ids_tensor_one_hot.grad.shape,
        # np.count_nonzero(token_ids_tensor_one_hot.detach().numpy(), axis=1))

        # Normalize the values so they add up to 1
        token_importance = token_importance_raw / torch.sum(token_importance_raw)
    elif aggregation == "sum":
        token_importance_raw = torch.sum(inputs_embeds.grad, dim=1)
        token_importance = token_importance_raw  # Hmmm, how to normalize if it includes negative values
    elif aggregation == "mean":
        token_importance_raw = torch.mean(inputs_embeds.grad, dim=1)
        token_importance = token_importance_raw  # Hmmm, how to normalize if it includes negative values

    inputs_embeds.grad.data.zero_()
    return token_importance


def gradient_x_inputs_attribution(prediction_logit, encoder_inputs_embeds, decoder_inputs_embeds: Optional = None,
                                  retain_graph=True) -> torch.Tensor:

    # only works in batches of 1
    assert len(encoder_inputs_embeds.shape) == 3 and encoder_inputs_embeds.shape[0] == 1
    if decoder_inputs_embeds is not None:
        assert len(decoder_inputs_embeds.shape) == 3 and decoder_inputs_embeds.shape[0] == 1
        decoder_inputs_embeds.retain_grad()
    encoder_inputs_embeds.retain_grad()

    # back-prop gradient
    prediction_logit.backward(retain_graph=retain_graph)
    decoder_grad = decoder_inputs_embeds.grad if decoder_inputs_embeds is not None else None
    encoder_grad = encoder_inputs_embeds.grad

    # Grad X Input
    grad_enc_x_input = encoder_grad * encoder_inputs_embeds

    if decoder_grad is not None:
        grad_dec_x_input = decoder_grad * decoder_inputs_embeds
        grad_enc_x_input = encoder_grad * encoder_inputs_embeds
        grad_x_input = torch.cat([grad_enc_x_input, grad_dec_x_input], dim=1)[0]
    else:
        grad_x_input = grad_enc_x_input[0]

    # Turn into a scalar value for each input token by taking L2 norm
    feature_importance = torch.norm(grad_x_input, dim=1)

    # Normalize so we can show scores as percentages
    token_importance_normalized = feature_importance / torch.sum(feature_importance)

    # Zero the gradient for the tensor so next backward() calls don't have
    # gradients accumulating
    if decoder_inputs_embeds is not None:
        decoder_inputs_embeds.grad.data.zero_()
    encoder_inputs_embeds.grad.data.zero_()

    return token_importance_normalized


def lm_model_forward(input_: torch.Tensor, decoder_: torch.Tensor, model, extra_forward_args: Dict[str, Any]) \
        -> torch.Tensor:
    if decoder_ is not None:
        output = model(inputs_embeds=input_, decoder_inputs_embeds=decoder_, **extra_forward_args)
    else:
        output = model(inputs_embeds=input_, **extra_forward_args)

    return F.softmax(output.logits[:, -1, :], dim=-1)


def compute_integrated_gradients_scores(model: torch.nn.Module, forward_kwargs: Dict[str, Any],
                                        prediction_id: torch.Tensor, aggregation: str = "L2") -> torch.Tensor:

    def normalize_attributes(attributes: torch.Tensor) -> torch.Tensor:
        # attributes has shape (batch, sequence size, embedding dim)
        attributes = attributes.squeeze(0)

        if aggregation == "L2":  # norm calculates a scalar value (L2 Norm)
            norm = torch.norm(attributes, dim=1)
            attributes = norm / torch.sum(norm) # Normalize the values so they add up to 1
        else:
            raise NotImplemented

        return attributes

    extra_forward_args = {k: v for k, v in forward_kwargs.items() if k not in ['inputs_embeds', 'decoder_inputs_embeds']}
    input_ = forward_kwargs.get('inputs_embeds')
    decoder_ = forward_kwargs.get('decoder_inputs_embeds')

    if decoder_ is None:
        forward_func = partial(lm_model_forward, decoder_=decoder_, model=model, extra_forward_args=extra_forward_args)
        inputs = input_
    else:
        forward_func = partial(lm_model_forward, model=model, extra_forward_args=extra_forward_args)
        inputs = tuple([input_, decoder_])

    ig = IntegratedGradients(forward_func=forward_func)
    attributions = ig.attribute(inputs, target=prediction_id)
     
    if decoder_ is not None:
        # Does it make sense to concatenate encoder and decoder attributions before normalization?
        # We assume that the encoder/decoder embeddings are the same
        return normalize_attributes(torch.cat(attributions, dim=1))
    else:
        return normalize_attributes(attributions)


def compute_saliency_scores(prediction_logit: torch.Tensor,
                            encoder_token_ids_tensor_one_hot: torch.Tensor,
                            encoder_inputs_embeds: torch.Tensor,
                            decoder_token_ids_tensor_one_hot: Optional[torch.Tensor] = None,
                            decoder_inputs_embeds: Optional[torch.Tensor] = None,
                            gradient_kwargs: Dict[str, Any] = {},
                            gradient_x_input_kwargs: Dict[str, Any] = {},
                            saliency_methods: Optional[List[str]] = ['grad_x_input', 'gradient']) \
        -> Dict[str, torch.Tensor]:

    results = {}

    if 'grad_x_input' in saliency_methods:
        results['grad_x_input'] = gradient_x_inputs_attribution(
            prediction_logit=prediction_logit,
            encoder_inputs_embeds=encoder_inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            **gradient_x_input_kwargs
        )


    if 'gradient' in saliency_methods:
        results['gradient'] = saliency(
            prediction_logit=prediction_logit,
            encoder_token_ids_tensor_one_hot=encoder_token_ids_tensor_one_hot,
            decoder_token_ids_tensor_one_hot=decoder_token_ids_tensor_one_hot,
            **gradient_kwargs
        )

    return results


def get_lime_tokenized_text(text: str) -> List[str]:
    # mimics tokenization done in LIME
    # copied from https://github.com/marcotcr/lime/blob/fd7eb2e6f760619c29fca0187c07b82157601b32/lime/lime_text.py#L114
    splitter = re.compile(r'(\W+)|$')
    return [s for s in splitter.split(text) if s and not splitter.match(s)]


# lm: ecco.LM (avoid circular import just for typing purposes)
def compute_lime_scores(lm, enc_tokens: List[str], dec_tokens: Optional[List[str]] = None,
                        num_samples: int = None, top_labels: int = 5) -> Tuple[np.array, str]:

    def classifier_fn(texts: List[str]) -> np.ndarray:

        # init batch size
        batch_size = 256

        # Get token probabilities for all texts, do it in batches
        probs = []
        while True:
            try:
                offset = 0
                while offset < len(texts):

                    # get predictions
                    encoder_input_ids, encoder_attention_mask, decoder_input_ids = \
                        lm._init_ids_and_attention_masks(texts[offset: offset + batch_size])
                    encoder_inputs_embeds, _ = lm._get_embeddings(encoder_input_ids)
                    decoder_inputs_embeds, _ = lm._get_embeddings(decoder_input_ids) \
                                                   if decoder_input_ids is not None else None, None
                    extra_forward_kwargs = {
                        'attention_mask': encoder_attention_mask,
                        'use_cache': False,
                        'return_dict': True,
                    }
                    probs.append(
                        lm_model_forward(
                            input_=encoder_inputs_embeds, decoder_=decoder_inputs_embeds,
                            model=lm.model, extra_forward_args=extra_forward_kwargs
                        ).cpu().detach().numpy()
                    )

                    # increment offset
                    offset += batch_size

            except RuntimeError:
                # tried to allocate too much memory, reduce batch size
                batch_size //= 2
                continue

            break # exit loop

        return np.concatenate(probs, axis=0)

    # pass tokens to string text
    text = "".join(enc_tokens if enc_tokens is not None else []).strip()
    text += "".join(dec_tokens if dec_tokens is not None else []).strip()

    # init LIME explainer
    # TODO: LimeTextExplainer only does input perturbations
    explainer = LimeTextExplainer(
        class_names=list(sorted(lm.tokenizer.vocab.items(), key=lambda item: item[1]))
    )

    if not num_samples:
        # heuristic to get maximum number of ordered combinations for text tokens
        # we set the number of samples to be the minimum between this value and 5000
        tokens = get_lime_tokenized_text(text)
        num_samples = len(set(str(sorted(comb)) for size in range(len(tokens)) for comb in combinations(tokens, size)))
        num_samples = min(num_samples, 5000)

    # get explainability object
    exp = explainer.explain_instance(text, classifier_fn=classifier_fn, num_samples=num_samples, top_labels=top_labels)

    # get token scores for top label
    top_label_weights = exp.as_map()[exp.top_labels[0]] # get weights for top label
    top_label_weights = torch.tensor([v for _, v in sorted(top_label_weights, key=lambda item: item[0])])
    top_label_scores = F.softmax(top_label_weights, dim=-1)

    # get extra html code
    html_code = exp.as_html()

    return top_label_scores, html_code
