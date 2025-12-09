from itertools import zip_longest
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, TokenClassifierOutput
from transformers.models.bert.modeling_bert import (
    # BERT_INPUTS_DOCSTRING,
    # BERT_START_DOCSTRING,
    BertEmbeddings,
    BertEncoder,
    BertOnlyMLMHead,
)
from transformers.utils import (
    # add_start_docstrings,
    # add_start_docstrings_to_model_forward,
    logging,
)

try:
    from transformers.utils.auto_docstring import auto_docstring
except ImportError:
    # For backward compatibility with transformers before the auto_docstring decorator
    def auto_docstring(custom_intro: str):
        def decorator(cls):
            return cls

        return decorator


from musicbert_hf import from_fairseq
from musicbert_hf.constants import INPUT_PAD

# MonkeyPatch: replace BertModel.forward with our version
from musicbert_hf.hf_monkeypatch import forward as hf_forward  # noqa: F401
from musicbert_hf.utils.misc import zip_longest_with_error

logger = logging.get_logger(__name__)


class CompoundEmbeddings(BertEmbeddings):
    """
    Closely modeled on `BertEmbeddings` class. We don't subclass because we don't
    want the `word_embeddings` layer.
    """

    def __init__(self, config, *, compound_ratio: int = 8, upsample: bool = True):
        super().__init__(config)
        self.fairseq_position_embeddings = config.fairseq_position_embeddings
        self.compound_ratio = compound_ratio
        self.downsampling = nn.Sequential(
            nn.Linear(config.hidden_size * compound_ratio, config.hidden_size)
        )
        self.upsample = upsample
        # (Malcolm 2023-09-07) if `self.upsample` is False, we won't ever use
        #   self.upsampling, but we nevertheless create it so that the model will
        #   match the pretrained checkpoints.
        self.upsampling = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * compound_ratio)
        )

        # Overwrite `position_embeddings` because we need to set padding_idx to match
        #   fairseq implementation
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=1
        )
        if hasattr(self, "token_type_ids"):
            # BertModel.forward (or the monkeypatched version) checks for
            #   a token_type_ids attribute and then does things we don't want
            #   if it finds it, so we make sure it doesn't exist.
            delattr(self, "token_type_ids")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # input_ids: batch * compound_seq
        assert input_ids is not None and input_ids.ndim == 2

        batch, compound_seq = input_ids.shape
        ratio = self.compound_ratio

        assert not past_key_values_length, (
            "past_key_values_length not supported for compound mode"
        )
        assert compound_seq % ratio == 0, (
            f"token sequences length should be multiple of {ratio} for compound mode"
        )

        seq = compound_seq // ratio
        # (Malcolm 2024-02-20) I'm not sure what the motivation for this assertion
        #   is, hidden states for intermediate layers seem to work as normal.
        # assert last_state_only, "hidden states not available for compound mode"
        assert position_ids is None, (
            "custom position_ids are not supported for compound mode"
        )

        # (Malcolm 2024-03-13) unlike the fairseq implementation, we give position
        #   ids to padding tokens but I don't think this should matter since we
        #   ignore them later in any case.
        # (Malcolm 2024-03-15) fairseq begins position ids from 2
        if self.fairseq_position_embeddings:
            position_ids = self.position_ids[:, 2 : seq + 2]
        else:
            position_ids = self.position_ids[:, :seq]

        assert inputs_embeds is None, (
            "inputs_embeds are not supported for compound mode"
        )
        assert (token_type_ids is None) or not token_type_ids.any(), (
            "token_type_ids are not supported for compound mode"
        )

        flat_embeds = self.word_embeddings(input_ids)
        unflattened = flat_embeds.view(
            flat_embeds.shape[0], flat_embeds.shape[1] // ratio, -1
        )
        embeddings = self.downsampling(unflattened)

        # Learned position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def possibly_upsample(self, x):
        if self.upsample:
            x = self.upsampling(x).view(
                x.shape[0], x.shape[1] * self.compound_ratio, -1
            )
        return x


class MusicBertEncoder(BertModel):
    def __init__(
        self,
        config,
        *,
        compound_ratio: int = 8,
        add_pooling_layer=False,
        upsample: bool = True,
    ):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.config = config
        self.embeddings = CompoundEmbeddings(config, upsample=upsample)
        self.encoder = BertEncoder(config)
        self.compound_ratio = compound_ratio

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if attention_mask is None:
            assert inputs_embeds is None
            assert input_ids is not None

            batch_size, compound_seq = input_ids.shape
            assert compound_seq % self.compound_ratio == 0, (
                f"token sequences length should be multiple of {self.compound_ratio} for compound mode"
            )
            seq = compound_seq // self.compound_ratio
            assert past_key_values is None
            attention_mask = torch.ones(((batch_size, seq)), device=input_ids.device)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            raise NotImplementedError
        else:
            output.last_hidden_state = self.embeddings.possibly_upsample(  # type:ignore
                output.last_hidden_state  # type:ignore
            )
        return output


class MusicBertConfig(BertConfig):
    def __init__(
        self,
        *args,
        tie_word_embeddings=False,
        fairseq_position_embeddings=False,
        **kwargs,
    ):
        # Make sure tie_word_embeddings is False by default since it
        #   is not implemented for MusicBert
        super().__init__(*args, tie_word_embeddings=False, **kwargs)
        # Fairseq position embeddings begin at 2, so we need to offset the indexing
        #   of the position ids by 2.
        self.fairseq_position_embeddings = fairseq_position_embeddings


@auto_docstring(custom_intro="""MusicBert model for MLM pre-training task.""")
class MusicBert(BertPreTrainedModel):
    # (Malcolm 2024-03-12) BertForMaskedLM defines the following attributes. However,
    #   tied weights are not implemented for MusicBert.
    # _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `MusicBert` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        assert not config.tie_word_embeddings, (
            "`tie_word_embeddings` is not implemented"
        )

        self.bert = MusicBertEncoder(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # In transformers v5, this decorator and the BERT_INPUTS_DOCSTRING
    #  seem to be gone, in favor (I guess) of auto_docstring used above.
    # @add_start_docstrings_to_model_forward(
    #     BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    # )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaSequenceTaggingHead(nn.Module):
    """Head for sequence tagging/token-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = from_fairseq.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        if q_noise != 0:
            raise NotImplementedError
        self.out_proj = nn.Linear(inner_dim, num_classes)
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError(
                    "Attempting to use Spectral Normalization with Quant Noise. "
                    "This is not officially supported"
                )
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        # TODO: (Malcolm 2023-09-05)
        # https://github.com/facebookresearch/fairseq/pull/1709/files#r381391530
        # Would it make sense to add layer_norm here just like in the RobertaLMHead?
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MusicBertTokenClassificationConfig(MusicBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = kwargs.get("num_labels", 1)


class MusicBertTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        if config.is_decoder:
            logger.warning(
                "If you want to use `MusicBert` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        assert not config.tie_word_embeddings, (
            "`tie_word_embeddings` is not implemented"
        )

        self.bert = MusicBertEncoder(config, add_pooling_layer=False, upsample=False)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        # self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = RobertaSequenceTaggingHead(
            input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            num_classes=config.num_labels,
            activation_fn=config.classifier_activation,
            pooler_dropout=classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def compute_loss(logits, labels, num_items_in_batch):
        if isinstance(logits, dict):
            # HuggingFace uses `TokenClassifierOutput` which is a dict subtype
            logits = logits["logits"]

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="mean"
        )

        # I'm not sure why we would want to divide cross-entropy by the number of
        #   elements; it doesn't grow with the number of elements assuming we
        #   apply reduction="mean"
        # if num_items_in_batch is None:
        #     num_items_in_batch = 1

        return loss  # / num_items_in_batch

    # @add_start_docstrings_to_model_forward(
    #     BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # QUESTION: (Malcolm 2024-03-16) do we want to add dropout here?
        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels, num_items_in_batch=labels.shape[0])

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaSequenceMultiTaggingHead(nn.Module):
    """Head for sequence tagging/token-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes: Sequence[int],
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__()
        sub_heads = []
        for n_class in num_classes:
            sub_heads.append(
                RobertaSequenceTaggingHead(
                    input_dim,
                    inner_dim,
                    n_class,
                    activation_fn,
                    pooler_dropout,
                    q_noise,
                    qn_block_size,
                    do_spectral_norm,
                )
            )
        self.n_heads = len(sub_heads)
        self.multi_tag_sub_heads = nn.ModuleList(sub_heads)

    def forward(self, features, **kwargs):
        x = [sub_head(features) for sub_head in self.multi_tag_sub_heads]
        return x


class RobertaSequenceConditionalMultiTaggingHead(RobertaSequenceMultiTaggingHead):
    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes: Sequence[int],
        activation_fn,
        pooler_dropout,
        q_noise=0,
        qn_block_size=8,
        do_spectral_norm=False,
    ):
        super().__init__(
            input_dim,
            inner_dim,
            num_classes,
            activation_fn,
            pooler_dropout,
            q_noise,
            qn_block_size,
            do_spectral_norm,
        )
        projections = []
        for n_class in num_classes[:-1]:
            projections.append(nn.Linear(input_dim + n_class, input_dim))
        self.projections = nn.ModuleList(projections)

    def forward(self, features, **kwargs):
        out = []
        for sub_head, proj in zip_longest(
            self.multi_tag_sub_heads, self.projections, fillvalue=None
        ):
            assert sub_head is not None
            logits = sub_head(features)
            out.append(logits)

            if proj is not None:
                features = proj(torch.concat((features, logits), dim=-1))

        return out


class MusicBertMultiTaskTokenClassificationConfig(MusicBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_multi_labels = kwargs.get("num_multi_labels", 1)
        self.chained_output_heads = kwargs.get("chained_output_heads", False)


class MusicBertMultiTaskTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_multi_labels
        self.num_tasks = len(self.num_labels)
        if config.is_decoder:
            logger.warning(
                "If you want to use `MusicBert` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        assert not config.tie_word_embeddings, (
            "`tie_word_embeddings` is not implemented"
        )

        self.bert = MusicBertEncoder(config, add_pooling_layer=False, upsample=False)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )

        if config.chained_output_heads:
            output_cls = RobertaSequenceConditionalMultiTaggingHead
        else:
            output_cls = RobertaSequenceMultiTaggingHead

        self.classifier = output_cls(
            input_dim=config.hidden_size,
            inner_dim=config.hidden_size,
            num_classes=config.num_multi_labels,
            activation_fn=config.classifier_activation,
            pooler_dropout=classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def compute_loss(logits, labels, num_items_in_batch):
        if isinstance(logits, dict):
            logits = logits["logits"]

        losses = []
        for these_logits, these_labels in zip_longest_with_error(logits, labels):
            this_loss = F.cross_entropy(
                these_logits.view(-1, these_logits.shape[-1]),
                these_labels.view(-1),
                reduction="mean",
            )
            losses.append(this_loss)

        # I'm not sure why we would want to divide cross-entropy by the number of
        #   elements; it doesn't grow with the number of elements assuming we
        #   apply reduction="mean"
        # if num_items_in_batch is None:
        #     num_items_in_batch = 1

        # loss = torch.mean(torch.stack(losses) / num_items_in_batch)

        loss = torch.stack(losses).mean()
        return loss

    # @add_start_docstrings_to_model_forward(
    #     BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    # )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] | list[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(num_tasks, batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`. Can also be a list of length `num_tasks` of tensors of shape `(batch_size, sequence_length)`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if isinstance(labels, torch.Tensor):
            assert labels.ndim == 3, (
                "labels must have shape (num_tasks, batch_size, sequence_length)"
            )
            assert labels.shape[0] == self.num_tasks, (
                "labels must have shape (num_tasks, batch_size, sequence_length)"
            )
        elif isinstance(labels, list):
            assert len(labels) == self.num_tasks, "labels must have length num_tasks"
            for label in labels:
                assert label.ndim == 2, (
                    "labels must have shape (batch_size, sequence_length)"
                )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # TODO: (Malcolm 2024-03-16) do we want to add dropout here?
        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            num_items_in_batch = input_ids.shape[0]
            loss = self.compute_loss(logits, labels, num_items_in_batch)

        if not return_dict:
            raise NotImplementedError
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,  # type:ignore
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MusicBertMultiTaskTokenClassConditionedConfig(MusicBertConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_multi_labels = kwargs.get("num_multi_labels", 1)
        self.chained_output_heads = kwargs.get("chained_output_heads", False)
        self.z_mlp_layers = kwargs.get("z_mlp_layers", 2)
        self.z_embed_dim = kwargs.get("z_embed_dim", 128)
        self.z_mlp_norm = kwargs.get("z_mlp_norm", "yes")
        self.z_vocab_size = kwargs.get("z_vocab_size", 10)
        # "concat" or "proj[ect]"
        self.z_combine_procedure = kwargs.get("z_combine_procedure", "concat")


ACTIVATIONS = {"gelu": nn.GELU}


def mlp_layer(input_dim, output_dim, dropout, activation_fn, norm=True):
    modules: List[nn.Module] = [nn.Linear(input_dim, output_dim)]
    if dropout:
        modules.append(nn.Dropout(dropout))
    if norm:
        modules.append(nn.LayerNorm(output_dim))
    if activation_fn is not None:
        modules.append(ACTIVATIONS[activation_fn]())
    return nn.Sequential(*modules)


class MLP(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        norm: bool,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        assert n_layers > 0
        layers = []
        for _ in range(n_layers - 1):
            layers.append(
                mlp_layer(hidden_dim, hidden_dim, dropout, activation_fn, norm)
            )
        layers.append(mlp_layer(hidden_dim, output_dim, dropout, activation_fn, norm))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        return self.layers(x)


class MusicBertMultiTaskTokenClassConditioned(BertPreTrainedModel):
    def __init__(self, config: MusicBertMultiTaskTokenClassConditionedConfig):
        super().__init__(config)
        self.num_labels = config.num_multi_labels
        self.num_tasks = len(self.num_labels)
        if config.is_decoder:
            logger.warning(
                "If you want to use `MusicBert` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        assert not config.tie_word_embeddings, (
            "`tie_word_embeddings` is not implemented"
        )

        self.bert = MusicBertEncoder(config, add_pooling_layer=False, upsample=False)

        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.z_encoder = MLP(
            n_layers=config.z_mlp_layers,
            vocab_size=config.z_vocab_size,
            output_dim=config.z_embed_dim,
            hidden_dim=config.z_embed_dim,
            dropout=config.hidden_dropout_prob,
            norm=config.z_mlp_norm == "yes",
        )

        if config.z_combine_procedure == "concat":
            self.combine_f = lambda x, z: torch.concat([x, z], dim=-1)
            self.output_dim = config.hidden_size + config.z_embed_dim
        elif config.z_combine_procedure[:4] == "proj":
            self.combine_projection = nn.Linear(
                config.z_embed_dim + config.hidden_size, config.hidden_size
            )

            def combine_f(x, z):
                return self.combine_projection(torch.concat([x, z], dim=-1))

            self.combine_f = combine_f
            self.output_dim = config.hidden_size
        else:
            raise ValueError

        if config.chained_output_heads:
            output_cls = RobertaSequenceConditionalMultiTaggingHead
        else:
            output_cls = RobertaSequenceMultiTaggingHead

        self.classifier = output_cls(
            input_dim=self.output_dim,
            # I'm not sure how deliberate it was, but in my FairSEQ
            # implementation, we use the same inner_dim as the input_dim.
            # There's a plausible case that it should instead be config.hidden_size,
            # although it probably doesn't matter too much.
            inner_dim=self.output_dim,
            num_classes=config.num_multi_labels,
            activation_fn=config.classifier_activation,
            pooler_dropout=classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def compute_loss(logits, labels, num_items_in_batch):
        if isinstance(logits, dict):
            logits = logits["logits"]

        losses = []
        for these_logits, these_labels in zip_longest_with_error(logits, labels):
            this_loss = F.cross_entropy(
                these_logits.view(-1, these_logits.shape[-1]),
                these_labels.view(-1),
                reduction="mean",
            )
            losses.append(this_loss)

        # I'm not sure why we would want to divide cross-entropy by the number of
        #   elements; it doesn't grow with the number of elements assuming we
        #   apply reduction="mean"
        # if num_items_in_batch is None:
        #     num_items_in_batch = 1

        # loss = torch.mean(torch.stack(losses) / num_items_in_batch)

        loss = torch.stack(losses).mean()
        return loss

    # @add_start_docstrings_to_model_forward(
    #     BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    # )
    def forward(
        self,
        conditioning_ids: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] | list[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if isinstance(labels, torch.Tensor):
            assert labels.ndim == 3, (
                "labels must have shape (num_tasks, batch_size, sequence_length)"
            )
            assert labels.shape[0] == self.num_tasks, (
                "labels must have shape (num_tasks, batch_size, sequence_length)"
            )
        elif isinstance(labels, list):
            assert len(labels) == self.num_tasks, "labels must have length num_tasks"
            for label in labels:
                assert label.ndim == 2, (
                    "labels must have shape (batch_size, sequence_length)"
                )
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        z = self.z_encoder(conditioning_ids)
        sequence_output = self.combine_f(sequence_output, z)

        # TODO: (Malcolm 2024-03-16) do we want to add dropout here?
        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            num_items_in_batch = input_ids.shape[0]
            loss = self.compute_loss(logits, labels, num_items_in_batch)

        if not return_dict:
            raise NotImplementedError
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,  # type:ignore
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


SHARED_PARAMS = dict(
    classifier_activation="gelu",
    pooler_activation_fn="tanh",
    hidden_dropout_prob=0.1,  # TODO confirm
    attention_probs_dropout_prob=0.1,
    # activation_dropout = 0.0,  # TODO confirm
    # pooler_dropout = 0.0, # TODO confirm
    # encoder_layers_to_keep = None,
    # encoder_layerdrop = 0.0,
    # untie_weights_roberta = False,
    # spectral_norm_classification_head = False,
    tie_word_embeddings=False,
    pad_token_id=INPUT_PAD,
    max_position_embeddings=2048,
)

BERT_PARAMS = {
    "base": dict(
        num_hidden_layers=12,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
    )
    | SHARED_PARAMS,
    "tiny": dict(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=512,
        num_attention_heads=2,
    )
    | SHARED_PARAMS,
}


def freeze_layers(model: nn.Module, layers: Sequence[int] | int | None):
    if layers is None:
        return model
    if isinstance(layers, int):
        layers = list(range(layers))
    for name, param in model.named_parameters():
        for layer in layers:
            if name.startswith(f"bert.encoder.layer.{layer}."):
                logger.debug(f"Freezing {name}")
                param.requires_grad = False
        if name.startswith("bert.embeddings"):
            logger.debug(f"Freezing {name}")
            # (Malcolm 2025-01-22) if we freeze any layers, we also freeze the
            # embeddings. Eventually we might want to freeze the embeddings separately.
            param.requires_grad = False
