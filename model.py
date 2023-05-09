import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertEmbeddings, BertEncoder

from category_id_map import CATEGORY_ID_LIST
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from masklm import MaskLM
from configparser import ConfigParser


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, out_size, linear_layer_size, hidden_dropout_prob, num_label):
        super().__init__()
        self.norm= nn.BatchNorm1d(out_size)
        self.dense = nn.Linear(out_size, linear_layer_size[0])
        self.norm_1= nn.BatchNorm1d(linear_layer_size[0])
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.dense_1 = nn.Linear(linear_layer_size[0], linear_layer_size[1])
        self.norm_2= nn.BatchNorm1d(linear_layer_size[1])
        self.out_proj = nn.Linear(linear_layer_size[1], num_label)

    def forward(self, features, **kwargs):
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        x = self.dropout(x)
        x = self.dense_1(x)
        x = torch.relu(self.norm_2(x))
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).\
            expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask

        return mean_embeddings


class MyBertModel(BertModel):
    def forward_embedding(
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device=self.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        return embedding_output


class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        self.lm = MaskLM(tokenizer_path=args.bert_dir)
        self.video_lm = MaskLM(tokenizer_path=args.bert_dir)
        self.bert_cfg = BertConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        # self.bert_cfg.vocab_size = 768

        self.bert_output_size = 768
        self.bert = MyBertModel.from_pretrained(
            args.bert_dir, cache_dir=args.bert_cache
        )

        # self.text_embedding = BertEmbeddings(self.bert_cfg)
        self.video_embedding = BertEmbeddings(self.bert_cfg)
        self.bert_encoder = BertEncoder(self.bert_cfg)

        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(
            channels=args.vlad_hidden_size, ratio=args.se_ratio)

        self.fit_linear = nn.Linear(1024, 32*768).to(args.device)
        self.fit_activate = nn.ReLU().to(args.device)

        self.last_meanpooling = MeanPooling().to(args.device)
        self.fusion = ConcatDenseSE(
            args.vlad_hidden_size + self.bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        self.cls = BertOnlyMLMHead(self.bert_cfg)
        self.newfc_hidden = torch.nn.Linear(21128, 512)
        # TODO:EDITED FLAG
        # self.classifier = nn.Linear(512, len(CATEGORY_ID_LIST))
        self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))
        self.cls_head = ClassificationHead(self.bert_cfg.vocab_size,
                                           linear_layer_size=[1024, 512],
                                           hidden_dropout_prob=0.2,
                                           num_label=len(CATEGORY_ID_LIST))
        # TODO: EDITED FLAG
        # self.newfit_linear = nn.Linear(768, 768)
        self.newfit_linear = nn.Linear(4096, 768)
        # self.fit_lm_linear = nn.Linear(self.bert_cfg.vocab_size, 1)
        

    def fit_dims(self, inputs):
        outputs = self.fit_linear(inputs)
        outputs = self.fit_activate(outputs)
        return outputs.reshape(-1, 32, 768)

    def forward(self, inputs, inference=False):
        if self.args.mlm:
            text_input_ids, text_lm_label = self.lm.torch_mask_tokens(inputs['title_input'].to('cpu'))
            text_input_ids = text_input_ids.to(self.device)
            text_lm_label = text_lm_label.to(self.device)

            text_embedding = self.bert.embeddings(
                text_input_ids, inputs['title_mask'])
        else:
            text_embedding = self.bert.embeddings(
                inputs['title_input'], inputs['title_mask']
            )
        # Note: text_embedding.shape(bs, 50, 768)

        # TODO: edited flag
        # vision_embedding = self.nextvlad(
        #     inputs['frame_input'], inputs['frame_mask'])

        # vision_embedding = self.enhance(vision_embedding)
#         fit_vision_embedding = self.fit_dims(vision_embedding)
        fit_vision_embedding = self.newfit_linear(inputs['frame_input'])# 缓解异质空间问题
        vision_bert_embedding = self.video_embedding(inputs_embeds=fit_vision_embedding)
        
        # all_embeddings = torch.cat([text_embedding, vision_bert_embedding], 1)
        # all_masks = torch.cat([inputs['title_mask'], inputs['frame_mask']], 1)
        all_embeddings = torch.cat([vision_bert_embedding], 1)
        all_masks = torch.cat([inputs['frame_mask']], 1)
        extened_attention_mask = self.bert.get_extended_attention_mask(
            all_masks, all_masks.size(), device=self.device)

        encoder_outputs = self.bert.encoder(all_embeddings,
                                            attention_mask=extened_attention_mask,
                                            use_cache=False,
                                            return_dict=True)
        sequence_output = encoder_outputs[0]    # last_hidden_states

        if self.args.mlm:
            # calculate masked mlm loss

            lm_prediction_scores = self.cls(sequence_output[:,:-inputs['frame_input'].size()[1],:]) # 只传入Text信息
            # lm_prediction_scores = self.fit_lm_linear(lm_prediction_scores)
            pred = lm_prediction_scores.contiguous().view(-1, self.bert_cfg.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, text_lm_label.view(-1)) / 1.25 / 3

        # [:, :-inputs['frame_input'].size()[1], :]# Cut Video Feature Part

#         pooled_output = self.bert.pooler(
#             sequence_output) if self.bert.pooler is not None else None

        # bert_embedding = BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )
        mean_embeddings = self.last_meanpooling(sequence_output, all_masks)
#         mean_embeddings = torch.einsum("bsh,bs,b->bh", sequence_output, all_masks.float(), 1 / all_masks.float().sum(dim=1) + 1e-9)

        # TODO: EDITED Flag
        # vision_embedding = self.enhance(vision_embedding)
        # final_embedding = self.fusion([vision_embedding, mean_embeddings])
        # prediction = self.classifier(final_embedding)

        # final_embedding = self.newfc_hidden(final_embedding)
#         prediction = self.classifier(final_embedding)
#         prediction = self.cls_head(mean_embeddings)
#         test_var = self.cls_head(final_embedding)
#         prediction = self.classifier(final_embedding)
        # prediction = lm_prediction_scores.contiguous().view(-1, len(CATEGORY_ID_LIST))
        prediction = self.classifier(mean_embeddings)
        if inference:
            return torch.argmax(prediction, dim=1)
        elif self.args.mlm:
            # training mode with mlm
            return self.cal_loss(prediction, inputs['label'], masked_lm_loss)
        else:
            # training mode without mlm
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label, masked_lm_loss=None):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        if masked_lm_loss is not None:
            return loss + masked_lm_loss, accuracy, pred_label_id, label
        else:
            return loss, accuracy, pred_label_id, label


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            args.bert_dir, cache_dir=args.bert_cache)
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(
            channels=args.vlad_hidden_size, ratio=args.se_ratio)
        bert_output_size = 768
        # bert_output_size=1024
        self.fusion = ConcatDenseSE(
            args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))

        # from skconv import SKConv
        # self.skmodule = SKConv(64, 1, 2, 1, 2)

    def forward(self, inputs, inference=False):
        # bert_embedding = self.bert(inputs['title_input'], inputs['title_mask'])[
        #     'pooler_output']

        vision_embedding = self.nextvlad(
            inputs['frame_input'], inputs['frame_mask'])
        vision_embedding = self.enhance(vision_embedding)

        # Note: vision_embedding.shape(bs, 1024)

        bert_embedding = self.bert(inputs['title_input'], inputs['title_mask'])[
            'pooler_output']

        final_embedding = self.fusion([vision_embedding, bert_embedding])
        # final_final_embedding = self.skmodule(final_embedding)
        prediction = self.classifier(final_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(
            self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(
            self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(
            self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape(
            [-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape(
            [-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape(
            [-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(
            in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(
            in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)
        # self.bert = BertModel.from_pretrained(args.bert_dir, cache=args.bert_cache)
        # self.bert = BertModel.from_pretrained('hfl/chinese-macbert-base', 'data/cache')
#         self.bert = BertModel.from_pretrained(
#             'hfl/chinese-roberta-wwm-ext', 'data/cache')

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding
