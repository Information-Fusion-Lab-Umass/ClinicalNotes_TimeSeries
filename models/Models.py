# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlyTextModel(nn.Module):
    # Model which only uses text part
    def __init__(self, device, BioBert, BioBertConfig):
        super(OnlyTextModel, self).__init__()
        text_embed_size =  BioBertConfig.hidden_size

        self.FinalFC = nn.Linear(text_embed_size, 1, bias=False)
        
        for p in self.FinalFC.parameters():
            nn.init.xavier_uniform_(p)
            
        self.final_act = torch.sigmoid
        self.criterion= nn.BCEWithLogitsLoss()
        self.device = device
        
        self.BioBert = BioBert
        
    def forward(self, text, attn):
        emb = self.BioBert(text, attn)
        emb = emb[0][:,0,:]
        logit_X = emb
        logits = self.FinalFC(logit_X)
        logits = logits.squeeze()
        probs = self.final_act(logits)
        return logits, probs
    
    def get_l2(self):
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.FinalFC.parameters():
            l2_reg += param.norm(2)
        return l2_reg

class TimeAttn(nn.Module):
    def __init__(self, AttnType="All"):
        super(TimeAttn, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.0))
        self.b = torch.nn.Parameter(torch.tensor(1.0))
        self.w1 = torch.nn.Parameter(torch.tensor(1/3))
        
        self.e = torch.nn.Parameter(torch.tensor(0.02))
        self.f = torch.nn.Parameter(torch.tensor(1.0))
        self.w2 = torch.nn.Parameter(torch.tensor(1/3))
        
        self.d = torch.nn.Parameter(torch.tensor(40.0))
        self.n = torch.nn.Parameter(torch.tensor(4.5))
        self.w3 = torch.nn.Parameter(torch.tensor(1/3))
        
        self.AttnType = AttnType
    def GetAgg(self, Vals, Times):
        # We are cutting times values at 1 hour so we don't run into NaNs below
        Times = torch.clamp(Times, 1)
        A_conv = 1/(self.a * (Times ** self.b))
        A_lin = torch.clamp(self.f - self.e * Times, 0)
        A_conc = 1/(1+(Times/self.d) ** self.n)
        if self.AttnType == "All":
            attns = self.w1 * A_conv + self.w2 * A_lin + self.w3 * A_conc
        attn_sum = torch.sum(attns)
        if attn_sum < 1e-5:
            attns = torch.ones_like(attns) / len(attns)
        else:
            attns = attns / attn_sum
        results = attns[None, :].mm(Vals).squeeze()
        return results
class ClinicalNotesModel(nn.Module):
    # Main model with which combines time-series and text part
    def __init__(self, dropout_keep_prob, W_embed, model_name, model_type, BioBert, notes_aggeregate, TSModel, TS_aggeregate, Attn_Type, device):
        super(ClinicalNotesModel, self).__init__()
        self.dropout_keep_prob = dropout_keep_prob
        if model_name != "BioBert":
            # Load word2vec weights for word embedding
            self.W_embed = nn.Embedding.from_pretrained(torch.tensor(W_embed, dtype=torch.float))
            # CNN model for time series the window sizes defined below
            self.sizes = range(2, 5)
            self.CNNs = nn.ModuleList([nn.Conv1d(200, 256, ngram_size, stride=1, padding=ngram_size//2) for ngram_size in self.sizes])
            # Dropout over the text embedding 
            self.EmbedDropout = nn.Dropout(1 - dropout_keep_prob)
        self.TSModel = TSModel
        self.TS_aggeregate = TS_aggeregate
        
        time_series_size = 256
        # LSTM model for time-series
        if TSModel == 'BiLSTM':
            self.lstm = nn.LSTM(76, time_series_size, 1, batch_first=True, bidirectional=True)        
        else:
            self.lstm = nn.LSTM(76, time_series_size, 1, batch_first=True)
            
        if TSModel == 'BiLSTM':
            time_series_size *= 2
            
        # Using attention
        if TS_aggeregate == 'Attention':
            self.TS_attn = Attention(time_series_size, Attn_Type)

        if model_name == 'avg_we':
            print('Using Average Embedding Vector')
            text_embed_size =  W_embed.shape[1]
        elif model_name == 'cnn':
            print('Using CNN for text part')
            text_embed_size =  len(self.sizes) * 256
        elif model_name == 'BioBert':
            print('Using BioBert model')
            text_embed_size =  BioBert.config.hidden_size
        
        if model_type == "baseline":
            print('Using only time-series')
            self.FinalFC = nn.Linear(time_series_size, 1, bias=False)
        elif model_type == "text_only":
            print('Using only text part')
            self.FinalFC = nn.Linear(text_embed_size, 1, bias=False)
        else:
            print('Using both text and time series')
            self.FinalFC = nn.Linear(time_series_size + text_embed_size, 1, bias=False)
        
        
        for p in self.FinalFC.parameters():
            # Using Xavier Uniform initialization for final fully connected layer
            nn.init.xavier_uniform_(p)
        self.final_act = torch.sigmoid
        self.model_name = model_name
        self.model_type = model_type
        self.criterion= nn.BCEWithLogitsLoss()
        self.device = device
        self.notes_aggeregate = notes_aggeregate
        
        if self.notes_aggeregate == 'TimeAttn':
            self.TA = TimeAttn(AttnType="All")
        elif self.notes_aggeregate == 'Attn':
            self.NotesAttn = Attention(text_embed_size, Attn_Type)
        
        self.BioBert = BioBert
    
    def forward(self, X, text, attns, times):
        # if model_type is baseline we only use the time-series part
        if self.model_type != 'baseline':
            if self.model_name != 'BioBert':
                # When not using bert based models, use word2vec embedding layer to get the embedding vectors
                embeds = self.W_embed(text)
            if self.model_name == 'avg_we':
                text_embeddings = torch.mean(embeds, axis=1)
            elif self.model_name == 'BioBert':
                if self.notes_aggeregate == 'Mean' or  self.notes_aggeregate == 'TimeAttn' or self.notes_aggeregate == 'Attn':
                    # For the models where we take the mean of embeddings generated for several notes we loop through the notes and take their mean
                    txt_arr = []
                    for txts, attn, ts in zip(text, attns, times):
                        if len(txts.shape) == 1:
                            # If there is only one note for a patient we just add a dimension
                            txts = txts[None, :]
                            attn = attn[None, :]
                        txtemb = self.BioBert(txts, attn)
                        emb = txtemb[0][:,0,:]
                        if self.notes_aggeregate == 'TimeAttn':
                            txt_arr.append(self.TA.GetAgg(emb, ts))
                        elif self.notes_aggeregate == 'Attn':
                            Attn_out, _ = self.NotesAttn(emb[-1,:].unsqueeze(0).unsqueeze(0), emb.unsqueeze(0))
                            txt_arr.append(Attn_out.squeeze())
                        else:
                            txt_arr.append(torch.mean(emb, axis=0))
                    text_embeddings = torch.stack(txt_arr)
                    # deleting some tensors to free up some space
                    del txt_arr
                else:
                    # If we are only using one note we just run bert based model and get its embedding
                    text_embeddings = self.BioBert(text, attns)
                    text_embeddings = text_embeddings[0][:,0,:]
            else:
                # If using CNN for texts
                result_tensors = []
                for i in range(len(self.sizes)):
                    # running CNN with different kernel sizes and concatenating the embeddings
                    # 256 -> 2,3 best yet.
                    text_conv1d = self.CNNs[i](embeds.transpose(1,2))
                    text_conv1d = text_conv1d.transpose(1, 2)
                    text_conv1d = F.relu(text_conv1d)
                    text_conv1d, _ = torch.max(text_conv1d, axis=1)
                    result_tensors.append(text_conv1d)
                text_embeddings = torch.cat(result_tensors, axis=1)
                text_embeddings = self.EmbedDropout(text_embeddings)
        # Getting the time-series part embedding
        rnn_outputs, _ = self.lstm(X)
        if self.TS_aggeregate == 'Last':
            mean_rnn_outputs = rnn_outputs[:, -1, :]
        elif self.TS_aggeregate == 'Mean':
            mean_rnn_outputs = torch.mean(rnn_outputs, 1)
        elif self.TS_aggeregate == 'Attention':
            mean_rnn_outputs, _ = self.TS_attn(rnn_outputs[:, -1, :].unsqueeze(1), rnn_outputs)
            mean_rnn_outputs = mean_rnn_outputs.squeeze(1)
        
        if self.model_type == 'baseline':
            logit_X = mean_rnn_outputs
        elif self.model_type == 'text_only':
            logit_X = text_embeddings
        else:
            # Concatenating time-series and text embedding
            logit_X = torch.cat((text_embeddings.float(), mean_rnn_outputs.float()), 1)
        
        #Final FC layer
        logits = self.FinalFC(logit_X)
        logits = logits.squeeze(dim=-1)
        probs = self.final_act(logits)
        return logits, probs
    
    def get_l2(self):
        # get l2 regularization weight of the cnn and final layer
        l2_reg = torch.tensor(0.).to(self.device)
        if self.model_name != 'BioBert':
            for cnn in self.CNNs:
                for param in cnn.parameters():
                    l2_reg += param.norm(2)
        for param in self.FinalFC.parameters():
            l2_reg += param.norm(2)
        return l2_reg
    
class Attention(nn.Module):
    # Attention from torchnlp
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights