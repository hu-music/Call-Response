import sys
import os
import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder, RecurrentDecoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note


D_MODEL = 512
N_LAYER = 6
N_HEAD = 8
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
TICK_RESOL = BEAT_RESOL // 4



def write_midi(words, path_outfile, word2event):

    class_keys = word2event.keys()
    # words = np.load(path_infile)
    midi_obj = miditoolkit.midi.parser.MidiFile()

    bar_cnt = 0
    cur_pos = 0

    all_notes = []

    cnt_error = 0
    for i in range(len(words)):
        vals = []
        for kidx, key in enumerate(class_keys):
            vals.append(word2event[key][words[i][kidx]])
        # print(vals)

        if vals[3] == 'Metrical':
            if vals[2] == 'Bar':
                bar_cnt += 1
            elif 'Beat' in vals[2]:
                beat_pos = int(vals[2].split('_')[1])
                cur_pos = bar_cnt * BAR_RESOL + beat_pos * TICK_RESOL

                # chord
                if vals[1] != 'CONTI' and vals[1] != 0:
                    midi_obj.markers.append(
                        Marker(text=str(vals[1]), time=cur_pos))

                if vals[0] != 'CONTI' and vals[0] != 0:
                    tempo = int(vals[0].split('_')[-1])
                    midi_obj.tempo_changes.append(
                        TempoChange(tempo=tempo, time=cur_pos))
            else:
                pass
        elif vals[3] == 'Note':

            try:
                pitch = vals[4].split('_')[-1]
                duration = vals[5].split('_')[-1]
                velocity = vals[6].split('_')[-1]

                if int(duration) == 0:
                    duration = 60
                end = cur_pos + int(duration)

                all_notes.append(
                    Note(
                        pitch=int(pitch),
                        start=cur_pos,
                        end=end,
                        velocity=int(velocity))
                    )
            except:
                continue
        else:
            pass

    # save midi
    piano_track = Instrument(0, is_drum=False, name='piano')
    piano_track.notes = all_notes
    midi_obj.instruments = [piano_track]
    midi_obj.dump(path_outfile)

################################################################################
# Sampling
################################################################################
# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    try:
        word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    except:
        word=sorted_index[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    try:
        word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    except:
        word=candi_index[0]
    return word


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


################################################################################
# Model
################################################################################


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('positional encoding:', self.pe[:, :x.size(1), :].shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class KnowledgeSelector(nn.Module):
    def __init__(self, hidden_size):
        super(KnowledgeSelector, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_state, knowledge_base):
        # Compute the similarity scores between the input state and each knowledge item
        scores = []
        for item in knowledge_base:
            item_state = knowledge_base[item]  #bs,length,hs
            item_score = F.cosine_similarity(self.linear(input_state), self.linear(item_state), dim=-1)
            scores.append(item_score)

        # Convert the scores to a probability distribution using softmax
        scores = F.softmax(torch.stack(scores, dim=0), dim=0)
        # Compute the weighted average of the knowledge items based on the scores
        output_state = input_state.clone()
        for i, item in enumerate(knowledge_base):
            output_state += scores[i].unsqueeze(-1) * knowledge_base[item]
        return output_state

class TransformerModel(nn.Module):
    def __init__(self, n_token, is_training=True):
        super(TransformerModel, self).__init__()

        # --- params config --- #
        self.n_token = n_token
        self.d_model = D_MODEL
        self.n_layer = N_LAYER #
        self.dropout = 0.1
        self.n_head = N_HEAD #
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [512, 256, 64, 32, 512, 128, 512]

        # --- modules config --- #
        # embeddings
        print('>>>>>:', self.n_token)
        self.word_emb_tempo     = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord     = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat   = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_type      = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_pitch     = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_duration  = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.word_emb_velocity  = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.pos_emb            = PositionalEncoding(self.d_model, self.dropout)

        # linear
        self.in_linear = nn.Linear(np.sum(self.emb_sizes), self.d_model)
        self.linear_knowledge = nn.Linear(self.d_model*2, self.d_model)
        self.knowledge_selector=KnowledgeSelector(self.d_model)
         # encoder
        if is_training:
            # encoder (training)
            self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="full",
            ).get()
            self.transformer_decoder=TransformerDecoderBuilder.from_kwargs(
                            n_layers=self.n_layer,
                            n_heads=self.n_head,
                            query_dimensions=self.d_model//self.n_head,
                            value_dimensions=self.d_model//self.n_head,
                            feed_forward_dimensions=2048,
                            activation='gelu',
                            dropout=0.1,
                            self_attention_type="full",
                            cross_attention_type="full"
                        ).get()
        else:
            # encoder (inference)
            self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                attention_type="full",
            ).get()
            self.transformer_decoder = RecurrentDecoderBuilder.from_kwargs(
                n_layers=self.n_layer,
                n_heads=self.n_head,
                query_dimensions=self.d_model//self.n_head,
                value_dimensions=self.d_model//self.n_head,
                feed_forward_dimensions=2048,
                activation='gelu',
                dropout=0.1,
                self_attention_type="full",
                cross_attention_type="full"
            ).get()
        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)

        # individual output
        self.proj_tempo    = nn.Linear(self.d_model, self.n_token[0])
        self.proj_chord    = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat  = nn.Linear(self.d_model, self.n_token[2])
        self.proj_type     = nn.Linear(self.d_model, self.n_token[3])
        self.proj_pitch    = nn.Linear(self.d_model, self.n_token[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[6])
    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss
    def compute_loss_class(self, predict, label):
        loss = self.loss_func(predict, label)
        # print('loss',loss)
        return torch.mean(loss)
    def train_step(self, x, target,label, src_mask,tgt_mask, loss_mask,knowledge_base):
        h  = self.forward_hidden(x,src_mask)
        if knowledge_base is not None:
            knowledge_info= self.knowledge_selector(h,knowledge_base)
            h = torch.cat([h,knowledge_info],dim=2)  # bs, length, d_model*2 if dim=2
            h= self.linear_knowledge(h)
        y_tempo, y_chord, y_barbeat,y_type, y_pitch, y_duration, y_velocity = self.forward_output(h, target,tgt_mask,label)

        # reshape (b, s, f) -> (b, f, s)
        y_tempo     = y_tempo[:, ...].permute(0, 2, 1)
        y_chord     = y_chord[:, ...].permute(0, 2, 1)
        y_barbeat   = y_barbeat[:, ...].permute(0, 2, 1)
        y_type      = y_type[:, ...].permute(0, 2, 1)
        y_pitch     = y_pitch[:, ...].permute(0, 2, 1)
        y_duration  = y_duration[:, ...].permute(0, 2, 1)
        y_velocity  = y_velocity[:, ...].permute(0, 2, 1)

        # loss
        loss_tempo = self.compute_loss(
                y_tempo, label[..., 0], loss_mask)
        loss_chord = self.compute_loss(
                y_chord, label[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(
                y_barbeat, label[..., 2], loss_mask)
        loss_type = self.compute_loss(
                y_type,  label[..., 3], loss_mask)
        loss_pitch = self.compute_loss(
                y_pitch, label[..., 4], loss_mask)
        loss_duration = self.compute_loss(
                y_duration, label[..., 5], loss_mask)
        loss_velocity = self.compute_loss(
                y_velocity, label[..., 6], loss_mask)

        return loss_tempo, loss_chord, loss_barbeat, loss_type, loss_pitch, loss_duration, loss_velocity

    def forward_hidden(self, x, src_mask):
        '''
        linear transformer: b x s x f
        x.shape=(bs, nf)
        '''

        # embeddings
        emb_tempo =    self.word_emb_tempo(x[..., 0])
        emb_chord =    self.word_emb_chord(x[..., 1])
        emb_barbeat =  self.word_emb_barbeat(x[..., 2])
        emb_type =     self.word_emb_type(x[..., 3])
        emb_pitch =    self.word_emb_pitch(x[..., 4])
        emb_duration = self.word_emb_duration(x[..., 5])
        emb_velocity = self.word_emb_velocity(x[..., 6])

        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_type,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)

        length_mask=LengthMask(torch.LongTensor(src_mask).to('cuda'),256)

        h = self.transformer_encoder(pos_emb,length_mask=length_mask) # y: b x s x d_model

        # y_type = self.proj_type(h)

        # project type
        return h

    def forward_output(self, h, y,tgt_mask,label):
        '''
        for training
        '''

        emb_tempo =    self.word_emb_tempo(y[..., 0])
        emb_chord =    self.word_emb_chord(y[..., 1])
        emb_barbeat =  self.word_emb_barbeat(y[..., 2])
        emb_type =     self.word_emb_type(y[..., 3])
        emb_pitch =    self.word_emb_pitch(y[..., 4])
        emb_duration = self.word_emb_duration(y[..., 5])
        emb_velocity = self.word_emb_velocity(y[..., 6])

        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_type,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)

        attn_mask = TriangularCausalMask(pos_emb.size(1), device=y.device)
        length_mask=LengthMask(torch.LongTensor(tgt_mask).to('cuda'),256)

        h = self.transformer_decoder(pos_emb,h,x_mask=attn_mask,x_length_mask=length_mask)
        y_type = self.proj_type(h)
        tf_skip_type = self.word_emb_type(label[..., 3])
        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_  = self.project_concat_type(y_concat_type)

        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)
        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)

        return  y_tempo, y_chord, y_barbeat,y_type,y_pitch, y_duration, y_velocity

    def forward_output_sampling(self, h, memory,y):
        '''
        for inference
        '''
        # sample type
        emb_tempo =    self.word_emb_tempo(y[..., 0])
        emb_chord =    self.word_emb_chord(y[..., 1])
        emb_barbeat =  self.word_emb_barbeat(y[..., 2])
        emb_type =     self.word_emb_type(y[..., 3])
        emb_pitch =    self.word_emb_pitch(y[..., 4])
        emb_duration = self.word_emb_duration(y[..., 5])
        emb_velocity = self.word_emb_velocity(y[..., 6])

        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_type,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)

        emb_linear = self.in_linear(embs)
        pos_emb = self.pos_emb(emb_linear)

        attn_mask = TriangularCausalMask(pos_emb.size(1), device=y.device)



        pos_emb = pos_emb.squeeze(0)
        y_,memory = self.transformer_decoder(pos_emb, h,memory_length_mask=attn_mask,state=memory) # y: s x d_model
        y_type = self.proj_type(y_)

        y_type_logit = y_type[0, :]
        cur_word_type = sampling(y_type_logit, p=0.90)

        type_word_t = torch.from_numpy(
                    np.array([cur_word_type])).long().cuda().unsqueeze(0)

        tf_skip_type = self.word_emb_type(type_word_t).squeeze(0)

        # concat
        y_concat_type = torch.cat([y_, tf_skip_type], dim=-1)
        y_  = self.project_concat_type(y_concat_type)

        # project other
        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)
        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)

        # sampling gen_cond
        cur_word_tempo =    sampling(y_tempo, t=1.2, p=0.9)
        cur_word_barbeat =  sampling(y_barbeat, t=1.2)
        cur_word_chord =    sampling(y_chord, p=0.99)
        # cur_word_type = sampling(y_type, p=0.90)
        cur_word_pitch =    sampling(y_pitch, p=0.9)
        cur_word_duration = sampling(y_duration, t=2, p=0.9)
        cur_word_velocity = sampling(y_velocity, t=5)

        # collect
        next_arr = np.array([
            cur_word_tempo,
            cur_word_chord,
            cur_word_barbeat,
            cur_word_type,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity,
            ])
        return next_arr,memory

    def inference(self, src,src_mask,dictionary,knowledge_base=None):
        event2word, word2event = dictionary
        classes = word2event.keys()

        def print_word_cp(cp):
            result = [word2event[k][cp[idx]] for idx, k in enumerate(classes)]

            for r in result:
                print('{:15s}'.format(str(r)), end=' | ')
            print('')

        init = np.array([
            [0, 0, 0, 3, 0, 0, 0], # start
        ])

        cnt_token = len(init)
        with torch.no_grad():
            final_res = []
            memory = None
            h = None

            cnt_bar = 1
            init_t = torch.from_numpy(init).long().unsqueeze(0).cuda()
            h = self.forward_hidden(
                    src,src_mask)

            if knowledge_base is not None:
                knowledge_info= self.knowledge_selector(h,knowledge_base)
                h = torch.cat([h,knowledge_info],dim=1)

            print('------ generate ------')
            input_ = init_t
            while(True):
                # sample others
                next_arr,memory = self.forward_output_sampling(h, memory,input_)
                final_res.append(next_arr[None, ...])
                print('bar:', cnt_bar, end= '  ==')
                print_word_cp(next_arr)

                # forward
                input_ = torch.from_numpy(next_arr).long().cuda()
                input_  = input_.unsqueeze(0).unsqueeze(0)


                # end of sequence
                if word2event['type'][next_arr[3]] == 'EOS':
                    break

                if word2event['bar-beat'][next_arr[2]] == 'Bar':
                    cnt_bar += 1

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res
