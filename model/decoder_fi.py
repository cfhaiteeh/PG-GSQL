""" Decoder for the SQL generation problem."""

from collections import namedtuple
import numpy as np
import math
import torch
import torch.nn.functional as F
from . import torch_utils
import gc
from .token_predictor import PredictionInput, PredictionInputWithSchema
import data_util.snippets as snippet_handler
from . import embedder
from data_util.vocabulary import EOS_TOK, UNK_TOK
from simplediff import diff
from . import add_util
from .attention import Attention, AttentionResult,AttentionWeight,AttentionAddMode
          
def flatten_distribution(distribution_map, probabilities):
    """ Flattens a probability distribution given a map of "unique" values.
        All values in distribution_map with the same value should get the sum
        of the probabilities.

        Arguments:
            distribution_map (list of str): List of values to get the probability for.
            probabilities (np.ndarray): Probabilities corresponding to the values in
                distribution_map.

        Returns:
            list, np.ndarray of the same size where probabilities for duplicates
                in distribution_map are given the sum of the probabilities in probabilities.
    """
    assert len(distribution_map) == len(probabilities)
    if len(distribution_map) != len(set(distribution_map)):
        idx_first_dup = 0
        seen_set = set()
        for i, tok in enumerate(distribution_map):
            if tok in seen_set:
                idx_first_dup = i
                break
            seen_set.add(tok)
        new_dist_map = distribution_map[:idx_first_dup] + list(
            set(distribution_map) - set(distribution_map[:idx_first_dup]))
        assert len(new_dist_map) == len(set(new_dist_map))
        new_probs = np.array(
            probabilities[:idx_first_dup] \
            + [0. for _ in range(len(set(distribution_map)) \
                                 - idx_first_dup)])
        assert len(new_probs) == len(new_dist_map)

        for i, token_name in enumerate(
                distribution_map[idx_first_dup:]):
            if token_name not in new_dist_map:
                new_dist_map.append(token_name)

            new_index = new_dist_map.index(token_name)
            new_probs[new_index] += probabilities[i +
                                                  idx_first_dup]
        new_probs = new_probs.tolist()
    else:
        new_dist_map = distribution_map
        new_probs = probabilities

    assert len(new_dist_map) == len(new_probs)

    return new_dist_map, new_probs

class SQLPrediction(namedtuple('SQLPrediction',
                               ('predictions',
                                'sequence',
                                'probability'))):
    """Contains prediction for a sequence."""
    __slots__ = ()

    def __str__(self):
        return str(self.probability) + "\t" + " ".join(self.sequence)

class SequencePredictorWithSchemaFi(torch.nn.Module):
    """ Predicts a sequence.

    Attributes:
        lstms (list of dy.RNNBuilder): The RNN used.
        token_predictor (TokenPredictor): Used to actually predict tokens.
    """
    def __init__(self,
                 params,
                 input_size,
                 output_embedder,
                 column_name_token_embedder,
                 token_predictor):
        super().__init__()

        schema_attention_key_size=300
        if params.use_query_attention:
            if params.discourse_level_lstm:
                self.in_size=300+50+300+300+schema_attention_key_size
            else:
                self.in_size=300+300+300+schema_attention_key_size
                
        else:
            if params.discourse_level_lstm:
                self.in_size=300+schema_attention_key_size+350
            else:
                self.in_size=300+schema_attention_key_size+300
        self.in_size-=300
        self.in_size-=50
        self.lstms = torch_utils.create_multilayer_lstm_params(params.decoder_num_layers,self.in_size ,300, "LSTM-d_fi")
        self.token_predictor = token_predictor
        self.output_embedder = output_embedder
        self.column_name_token_embedder = column_name_token_embedder
        self.start_token_embedding = torch_utils.add_params((300,), "y-0_fi")

        if params.mode_type=='sparc':
            self.voc=['UNK', 'EOS', 'select', 'from', 'value', ')', '(', 'where', '=', 'by', ',', 'count', 'group', 'order', 'limit', 'desc', '>', 'distinct', 'and', 'avg', 'having', '<', 'in', 'sum', 'max', 'asc', 'not', 'or', 'like', 'min', 'intersect', 'except', '!', 'union', 'between', '-', '+']
        else:
            self.voc=['UNK', 'EOS', 'from', 'select', 'value', '=', 'where', ')', '(', 'by', ',', 'count', 'group', 'order', 'distinct', 'and', 'desc', 'limit', '>', 'avg', 'having', 'max', 'in', '<', 'sum', 'intersect', 'not', 'min', 'asc', 'or', 'except', 'like', '!', 'union', 'between', '-', '+', 'ref_company_types', '/']
        self.input_size = input_size
        self.params = params
    def _initialize_decoder_lstm(self, encoder_state):
        decoder_lstm_states = []
        for i, _ in enumerate(self.lstms):
            encoder_layer_num = 0
            if len(encoder_state[0]) > 1:
                encoder_layer_num = i
            c_0 = encoder_state[0][encoder_layer_num].view(1,-1)
            h_0 = encoder_state[1][encoder_layer_num].view(1,-1)

            decoder_lstm_states.append((h_0, c_0))
        return decoder_lstm_states

    def get_output_token_embedding(self, output_token, input_schema,previous_querey_now,previous_query_state_now,cal_idx):
     
        if input_schema:
            if output_token in self.voc:
                output_token_embedding = self.output_embedder(output_token)
            else:
                output_token_embedding = input_schema.column_name_embedder(output_token, surface_form=True)
        else:
            output_token_embedding = self.output_embedder(output_token)
        return output_token_embedding

    def get_decoder_input(self, output_token_embedding, prediction):
        decoder_input = torch.cat([output_token_embedding, prediction.aug], dim=0)

        return decoder_input

    def forward(self,
                final_encoder_state,
                flat_sequence,sim_all_tks_list,sim_all_list,
                encoder_states,
                schema_states,
                previous_queries,
                previous_query_states,
                gold_match_id,
                info_list,
                max_generation_length,
                snippets=None,
                gold_sequence=None,
                input_sequence=None,
                input_schema=None,
                dropout_amount=0.,epoch=-1000,from_emb=None,status=0,bf_dict=None):
        """ Generates a sequence. """
        index = 0
        context_vector_size = self.input_size - self.params.output_embedding_size
        predictions = []
        sequence = []
        probability = 1.
        decoder_states = self._initialize_decoder_lstm(final_encoder_state)
        if self.start_token_embedding.is_cuda:
            decoder_input = torch.cat([self.start_token_embedding, torch.cuda.FloatTensor(self.in_size-300).fill_(0)], dim=0) 
        else:
            decoder_input = torch.cat([self.start_token_embedding, torch.zeros(context_vector_size)], dim=0)
        continue_generating = True
        flag=0
        tables=[]
        allTable=[x.lower() for x in input_schema.table_names_original]
        IdxCol=input_schema.column_names_original
        allCol=[]
        for idxc in IdxCol:
            allCol.append(idxc[1].lower())
        tabok=0
        tbidx=-1
        table_emb=[]
        table_tks=[]
        table_tk2id=input_schema.column_names_surface_form_to_id
        vis_t_e=[]
        for idx,tb in enumerate(allTable):
            if tb not in table_tk2id:
                continue
            id=table_tk2id[tb]
            if id in vis_t_e:
                continue
            vis_t_e.append(id)
            table_emb.append(schema_states[id])
            table_tks.append(tb)
        table_id=[]
        vis_tab=[]
        now_dict=dict()
        foreign_keys=input_schema.foreign_keys
        ocols_tb = [x[0] for x in input_schema.column_names_original]
        col_emb=[]
        col_tks=[]
        next_cols=[]
        
        switch_list=[]
        query_attention_weight_list=[]
        query_tks_list=[]


        gen_switch_list=[]
        question_attention_weight_list=[]
        cur_input_tks_list=[]
        query_attention_results_dis=[]
        previous_query_now=[]
        previous_query_state_now=[]
        while continue_generating:
            torch.cuda.empty_cache()
            if len(sequence) == 0 or sequence[-1] != EOS_TOK:
                _, decoder_state, decoder_states = torch_utils.forward_one_multilayer(self.lstms, decoder_input, decoder_states, dropout_amount)
                if (flag==1 or len(table_id)==0) :
                    fo_tb=[]
                    input_tab_emb=[]
                    input_tab_tks=[]
                    for k1,k2 in foreign_keys:
                        tbid1=ocols_tb[k1]
                        tbid2=ocols_tb[k2]
                        tb_1=allTable[tbid1]
                        tb_2=allTable[tbid2]
                        if tb_1 in vis_tab or tb_2 in vis_tab:
                            fo_tb.append(tb_1)
                            fo_tb.append(tb_2)

                    for tbtks,embs in zip(table_tks,table_emb):
                        if tbtks in vis_tab:
                            continue
                        if len(vis_tab)>0 and tbtks not in fo_tb:
                            continue
                        input_tab_emb.append(embs)
                        input_tab_tks.append(tbtks)
                    schema_v=schema_states
                    cur_input_tks_list.append([_ for _ in input_tab_tks])
                    prediction_input = PredictionInputWithSchema(decoder_state=decoder_state,
                                                             input_hidden_states=encoder_states,
                                                             schema_states=input_tab_emb,
                                                             previous_queries=previous_queries,
                                                             previous_query_states=previous_query_states,
                                                             snippets=snippets,
                                                             input_sequence=input_sequence,
                                                             input_schema=input_schema,
                                                             tks=input_tab_tks,flag=1,schema_v=schema_v
                                                             )
                
                else:
                    next_cols=[]
                    col_emb=[]
                    col_tks=[]
                    col_tk2id=input_schema.column_names_surface_form_to_id
                    col_emb.append(schema_states[col_tk2id['*']])
                    col_tks.append('*')
                    vis_e_c=[]
                    for tid,col in IdxCol:
                        if tid in table_id:
                            if col.lower() not  in col_tk2id:
                                continue
                            idx=col_tk2id[col.lower()]
                            if idx in vis_e_c:
                                continue
                            vis_e_c.append(idx)
                            col_emb.append(schema_states[idx])
                            col_tks.append(col.lower())
                    schema_v=schema_states

                    cur_input_tks_list.append([_ for _ in col_tks])
                    
                    prediction_input = PredictionInputWithSchema(decoder_state=decoder_state,
                                                                input_hidden_states=encoder_states,
                                                                schema_states=col_emb,
                                                                previous_queries=previous_queries,
                                                                 previous_query_states=previous_query_states,
                                                                snippets=snippets,
                                                                input_sequence=input_sequence,
                                                                input_schema=input_schema,
                                                                tks=col_tks,flag=0,schema_v=schema_v
                                                                )
                prediction = self.token_predictor(tabok,tbidx,tables,allTable,IdxCol,prediction_input,next_cols,previous_query_now,previous_query_state_now,decoder_input,input_schema.column_names_surface_form_to_id,self.output_embedder,query_attention_results_dis, dropout_amount=dropout_amount)
                predictions.append(prediction)
     
                if prediction.switch is not None:
                    switch_list.append(prediction.switch)
                    query_attention_weight_list.append(prediction.query_attention_weight)
                    query_tks_list.append(previous_query_now)

                else:
                    switch_list.append(None)
                    query_attention_weight_list.append([])
                    query_tks_list.append([])

                gen_switch_list.append(prediction.gen_switch)
                question_attention_weight_list.append(prediction.question_attention_weight)
                if gold_sequence:
                    output_token = gold_sequence[index]
                    cal_idx=gold_match_id[index]
                    output_token_embedding = self.get_output_token_embedding(output_token, input_schema,previous_query_now,previous_query_state_now,cal_idx)
                    decoder_input = self.get_decoder_input(output_token_embedding, prediction)
                    sequence.append(gold_sequence[index])
                    flag,table_id,vis_tab=add_util.judge_is_table(output_token,allTable,flag,table_id,vis_tab )
                    if index >= len(gold_sequence) - 1:
                        continue_generating = False
                else:

                  
                    from_switch = prediction.switch
                    if from_switch is not None:
                        now_probabilities = F.softmax(prediction.scores, dim=0)*from_switch
                        now_token_distribution_map = prediction.aligned_tokens

                        assert len(now_probabilities) == len(now_token_distribution_map)
                        
                        query_attention_weight=prediction.query_attention_weight.squeeze()*(1.0-from_switch)
                        
                        for qi,tp in enumerate(previous_query_now):
                            for idx,z in enumerate(prediction.aligned_tokens):
                                if z==tp:
                                    now_probabilities[idx]+=query_attention_weight[qi]
                        probabilities=now_probabilities.cpu().data.numpy().tolist()
                        distribution_map =now_token_distribution_map
                        assert len(probabilities) == len(distribution_map)

                    else:
                        probabilities= F.softmax(prediction.scores, dim=0).cpu().data.numpy().tolist()
                        distribution_map = prediction.aligned_tokens

                        assert len(probabilities) == len(distribution_map)
                        distribution_map, probabilities = flatten_distribution(distribution_map, probabilities)
                    distribution_map, probabilities = flatten_distribution(distribution_map, probabilities)
                    probabilities[distribution_map.index(UNK_TOK)] = 0.
                    argmax_index = int(np.argmax(probabilities))
                    argmax_token = distribution_map[argmax_index]
                    argmax_token,argmax_index=add_util.common_error_revise(argmax_token,argmax_index,sequence,input_sequence,previous_queries,probabilities,distribution_map,allCol)
                    sequence.append(argmax_token )
                    cal_idx=-1
                    if len(previous_query_states) > 0:
                        gold_match_id=cal_idx_now(previous_queries[-1],sequence)
                        cal_idx=gold_match_id[-1]
                    flag,table_id,vis_tab=add_util.judge_is_table(argmax_token,allTable,flag,table_id,vis_tab)
                    output_token_embedding = self.get_output_token_embedding(argmax_token, input_schema,previous_query_now,previous_query_state_now,cal_idx)
                    decoder_input = self.get_decoder_input(output_token_embedding, prediction)
                    probability *= probabilities[argmax_index]
                    continue_generating = False
                    if index < max_generation_length and argmax_token != EOS_TOK:
                        continue_generating = True
            index += 1
        if gold_sequence==None:
            sequence=add_util.key_gen(sequence,input_schema)


            if sequence[-3]=='between':
                sequence=sequence[:-3]
                sequence.extend(['between','1','and','1','EOS'])

        return SQLPrediction(predictions,
                             sequence,
                             probability),now_dict,switch_list,query_attention_weight_list,query_tks_list,gen_switch_list,question_attention_weight_list,cur_input_tks_list


def cal_idx_now(previous_sql,gold_sql):
    pre_cur_id=-1
    gold_match_id=[]
    diff_ans=diff(previous_sql,gold_sql)
    for x,y in diff_ans:
        if x=='=':
            for _ in range(len(y)):
                pre_cur_id+=1
                gold_match_id.append(pre_cur_id)
        elif x=='+':
            for _ in range(len(y)):
                gold_match_id.append(-1)
        else:
            for _ in range(len(y)):
                pre_cur_id+=1
    assert len(gold_match_id)==len(gold_sql)
    return gold_match_id