import torch
import torch.nn as nn
from typing import Literal
import math
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

class BertPooler(nn.Module):
    def __init__(self, config, pooling_fn):
        super().__init__()
        self.pooling_fn = pooling_fn
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled_output = self.pooling_fn(hidden_states, attention_mask)
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class BertForSequenceClassification(PreTrainedModel):
    
    def __init__(self, base_model: nn.Module, num_labels: int = 2, 
                 pooling_strategy: Literal["CLS", "AVG", "MUL_AVG"] = "CLS"):
        super().__init__(config=base_model.config)
        self.pooling_fn = self.__cont_embedding_factory[pooling_strategy]
        self.num_labels = num_labels
        self.bert = base_model.base_model
        self.config = base_model.config
        self.pooler = BertPooler(config=self.config, pooling_fn=self.pooling_fn)
        self.dropout = base_model.dropout
        self.classifier = base_model.classifier
        
        return None

    __cont_embedding_factory = {"CLS": lambda x, _: x.hidden_states[-1][:, 0, :],
                          "AVG": lambda x, attention_mask: SpecialBertForSequenceClassification.mean_pooling(x.hidden_states[-1], attention_mask),
                          "MUL_AVG": lambda x, attention_mask: SpecialBertForSequenceClassification.first_last_pooling(x, attention_mask)}

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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        pooled_output = self.pooler(outputs, attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @classmethod
    def mean_pooling(cls, output, attention_mask):
        token_embeddings = output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @classmethod
    def first_last_pooling(cls, output, attention_mask):
        batched = zip(*[torch.split(cls.mean_pooling(x, attention_mask), 1) for x in [output.hidden_states[0], output.hidden_states[-1]]])
        averaged_layers = torch.cat([torch.cat(layers, dim=0).mean(dim=0, keepdim=True) \
                                  for layers in batched], dim=0)
        return averaged_layers
