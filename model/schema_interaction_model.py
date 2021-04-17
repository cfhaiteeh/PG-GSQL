""" Class for the Sequence to sequence model for ATIS."""

import torch
import torch.nn.functional as F
from . import torch_utils
from .encoder import Encoder1
import data_util.snippets as snippet_handler
import data_util.sql_util
import data_util.vocabulary as vocab
from data_util.vocabulary import EOS_TOK, UNK_TOK
import data_util.tokenizers
from . import utils_bert
import numpy as np
from .token_predictor import construct_token_predictor
from .token_predictor_fi import construct_token_predictor_fi
from simplediff import diff
import random
from .attention import Attention
from .model import ATISModel, encode_snippets_with_states, get_token_indices
from data_util.utterance import ANON_INPUT_KEY
import model.add_util as au
import model.net_utils as nu
import torch.nn as nn

from .encoder import Encoder
from .decoder import SequencePredictorWithSchema
from .decoder_fi import SequencePredictorWithSchemaFi
import torch.nn.functional as F


import data_util.atis_batch

LIMITED_INTERACTIONS = {"raw/atis2/12-1.1/ATIS2/TEXT/TRAIN/SRI/QS0/1": 22,
                        "raw/atis3/17-1.1/ATIS3/SP_TRN/MIT/8K7/5": 14,
                        "raw/atis2/12-1.1/ATIS2/TEXT/TEST/NOV92/770/5": -1}

END_OF_INTERACTION = {"quit", "exit", "done"}

import copy



class SchemaInteractionATISModel(ATISModel):
    """ Interaction ATIS model, where an interaction is processed all at once.
    """

    def __init__(self,
                 params,
                 input_vocabulary,
                 output_vocabulary,
                 output_vocabulary_schema,
                 anonymizer):
        ATISModel.__init__(
            self,
            params,
            input_vocabulary,
            output_vocabulary,
            output_vocabulary_schema,
            anonymizer)
        self.schema_encoder = Encoder1(1, 768, 300,name1='schema_encoderl',name2='schema_encoderr')
        self.token_predictor = construct_token_predictor(params,
                                                         output_vocabulary,
                                                         self.utterance_attention_key_size,
                                                         self.schema_attention_key_size,
                                                         self.final_snippet_size,
                                                        
                                                         anonymizer)
        decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size
        self.decoder = SequencePredictorWithSchema(params, decoder_input_size, self.output_embedder,  None  , None, self.token_predictor)
        if params.mode_type=='sparc':
            self.voc=['UNK', 'EOS', 'select', 'from', 'value', ')', '(', 'where', '=', 'by', ',', 'count', 'group', 'order', 'limit', 'desc', '>', 'distinct', 'and', 'avg', 'having', '<', 'in', 'sum', 'max', 'asc', 'not', 'or', 'like', 'min', 'intersect', 'except', '!', 'union', 'between', '-', '+']
        else:
            self.voc=['UNK', 'EOS', 'from', 'select', 'value', '=', 'where', ')', '(', 'by', ',', 'count', 'group', 'order', 'distinct', 'and', 'desc', 'limit', '>', 'avg', 'having', 'max', 'in', '<', 'sum', 'intersect', 'not', 'min', 'asc', 'or', 'except', 'like', '!', 'union', 'between', '-', '+', 'ref_company_types', '/' ]
         
     
    def predict_turn(self,
                     utterance_final_state,
                     flat_sequence,
                     sim_all_list,sim_all_tks_list,
                     input_hidden_states,
                     schema_states,
                     previous_queries,
                     previous_query_states,
                     previous_final_state,
                     gold_match_id,
                     info_list,
                     max_generation_length,
                     gold_query=None,
                     snippets=None,
                     input_sequence=None,
                     input_sequences=None,
                     input_schema=None,
                     feed_gold_tokens=False,
                     training=False,epoch=-1000,from_emb=None,cmp_pre=None,bf_dict=None):
        """ Gets a prediction for a single turn -- calls decoder and updates loss, etc.

        TODO:  this can probably be split into two methods, one that just predicts
            and another that computes the loss.
        """
        predicted_sequence = []
        loss = None
        token_accuracy = 0.
        is_corr=0
        if feed_gold_tokens:
            decoder_results,now_dict,switch_list,query_attention_weight_list,query_tks_list ,question_attention_weight_list,cur_input_tks_list= self.decoder(utterance_final_state,
            flat_sequence,sim_all_tks_list,sim_all_list,
                                        input_hidden_states,
                                        schema_states,
                                        previous_queries,
                                        previous_query_states,
                                        previous_final_state,
                                        gold_match_id,
                                        info_list,
                                        max_generation_length,
                                        gold_sequence=gold_query,
                                        input_sequence=input_sequence,
                                        input_schema=input_schema,
                                        snippets=snippets,
                                        dropout_amount=self.dropout,epoch=epoch,from_emb=from_emb,bf_dict=bf_dict)
            
            all_scores = []
            all_alignments = []
            pre_len=-1
            if len(previous_queries) > 0:
                pre_len=len(previous_queries[-1])
            assert list(question_attention_weight_list[0].size())[1]==len(flat_sequence)
           
            for switch,prediction,query_attention_weight,query_tks,cur_input_tks in zip(switch_list, decoder_results.predictions,query_attention_weight_list,query_tks_list,
                                                                                                                    cur_input_tks_list):

                assert len(cur_input_tks)==len( prediction.aligned_tokens)-37
                scores= F.softmax(prediction.scores, dim=0)

                if switch is not None:
                    scores = scores*switch
                    query_attention_weight=query_attention_weight*(1.0-switch)

                    for qi,tp in enumerate(query_tks):
                        for idx,z in enumerate(prediction.aligned_tokens):
                            if z==tp:
                                scores[idx]+=query_attention_weight[qi]
                alignments = prediction.aligned_tokens



                all_scores.append(scores)
                all_alignments.append(alignments)


            gold_sequence = gold_query
          
            loss = torch_utils.compute_loss(gold_sequence, all_scores, all_alignments, get_token_indices,gold_match_id,pre_len)

            if not training:
                predicted_sequence = torch_utils.get_seq_from_scores(all_scores, all_alignments)

                token_accuracy = torch_utils.per_token_accuracy(gold_sequence, predicted_sequence)

        else:
            decoder_results,now_dict,switch_list,query_attention_weight_list,query_tks_list ,question_attention_weight_list,cur_input_tks_list= self.decoder(utterance_final_state,
            flat_sequence,sim_all_tks_list,sim_all_list,
                                        input_hidden_states,
                                        schema_states,
                                        previous_queries,
                                        previous_query_states,
                                        previous_final_state,
                                        gold_match_id,
                                        info_list,
                                        max_generation_length,
                                        input_sequence=input_sequence,
                                        input_sequences=input_sequences,
                                        input_schema=input_schema,
                                        snippets=snippets,
                                        dropout_amount=self.dropout,epoch=epoch,from_emb=from_emb,bf_dict=bf_dict)
            predicted_sequence = decoder_results.sequence
        decoder_states = [pred.decoder_state for pred in decoder_results.predictions]

        # fed_sequence contains EOS, which we don't need when encoding snippets.
        # also ignore the first state, as it contains the BEG encoding.


        return (predicted_sequence,
                loss,
                token_accuracy,
                decoder_states,
                decoder_results),now_dict,is_corr



    
    def encode_schema_bow_simple(self, input_schema):
        
        schema_states = []
        for column_name in input_schema.column_names_embedder_input:
            schema_states.append(input_schema.column_name_embedder_bow(column_name, surface_form=False, column_name_token_embedder=self.column_name_token_embedder))
        input_schema.set_column_name_embeddings(schema_states)
        return schema_states
    def get_input_embeding(self,utterance_toks,utterance_types,utterance_token_embedder):
        EMB=[]
        for tok,type in zip(utterance_toks,utterance_types):
            EMB.append(self.getTypeE(tok,type))
        return EMB    
    def get_all_bert_encoding(self,input_sequence, input_schema,dropout,max_seq_length=512):
        utterance_states, schema_token_states = utils_bert.get_bert_encoding(self.bert_config, self.model_bert, self.tokenizer, input_sequence, input_schema, bert_input_version=self.params.bert_input_version, num_out_layers_n=1, num_out_layers_h=1,max_seq_length=max_seq_length)
        raw_utterance_states=[_ for _ in utterance_states]
        schema_states=[]
        for schema_token_states1 in schema_token_states:
            if dropout:
                final_schema_state_one, _ = self.schema_encoder(schema_token_states1, lambda x: x, dropout_amount=self.dropout)
            else:
                final_schema_state_one, _ = self.schema_encoder(schema_token_states1, lambda x: x)
            schema_states.append(final_schema_state_one[1][-1])
        

        input_schema.set_column_name_embeddings(schema_states)

        return raw_utterance_states,  schema_states
    def get_bert_encoding(self, input_sequence, input_schema, discourse_state,idx_list,now_dict, dropout,max_seq_length=512,now_position=0):
        utterance_states, schema_token_states = utils_bert.get_bert_encoding(self.bert_config, self.model_bert, self.tokenizer, input_sequence, input_schema, bert_input_version=self.params.bert_input_version, num_out_layers_n=1, num_out_layers_h=1,max_seq_length=max_seq_length)
        if self.params.discourse_level_lstm :
            utterance_token_embedder = lambda x: torch.cat([x, discourse_state], dim=0)
        else:
            utterance_token_embedder = lambda x: x
        if dropout:
            final_utterance_state, utterance_states = self.bert_utterance_encoder(
            utterance_states,
                utterance_token_embedder,
                dropout_amount=self.dropout)
        else:
            final_utterance_state, utterance_states = self.bert_utterance_encoder(
                utterance_states,
                utterance_token_embedder)
        schema_states=[]
        indexofcl=input_schema.indexofcl
        new_schema_token_states=[]
        new_schema_token_states.append(schema_token_states[0])
        schema_token_states=schema_token_states[1:]


        for cl,ty in zip(input_schema.column_names_embedder_input,input_schema.column_type):
           
            if ty=='column' and cl !='column *':
                il=indexofcl[cl][0]
                rl=indexofcl[cl][1]
                zi=indexofcl[cl][2]
                cur_state=schema_token_states[zi][il:rl+1]
                new_schema_token_states.append(cur_state)
        new_schema_token_states.extend([_ for _ in schema_token_states])
        schema_token_states=new_schema_token_states
        for schema_token_states1 in schema_token_states:
            if dropout:
                final_schema_state_one, _,_,_ = self.schema_encoder(schema_token_states1, lambda x: x, dropout_amount=self.dropout)
            else:
                final_schema_state_one,_,_,_ = self.schema_encoder(schema_token_states1, lambda x: x)
            schema_states.append(final_schema_state_one[1][-1])
                    
     

        sim_all_list=[]
        sim_all_tks_list=[]
       

        input_schema.set_column_name_embeddings(schema_states)

        return final_utterance_state, utterance_states, schema_states,sim_all_list,sim_all_tks_list

    def get_query_token_embedding(self, output_token, input_schema):
        if input_schema:
            if not (output_token in self.voc or output_token  in input_schema.column_names_surface_form):
                output_token = 'distinct'
            if output_token in self.voc:
                output_token_embedding = self.output_embedder(output_token).cuda()
            else:
                output_token_embedding = input_schema.column_name_embedder(output_token, surface_form=True).cuda()
        else:
            output_token_embedding = self.output_embedder(output_token).cuda()
        return output_token_embedding

    def get_previous_queries(self, previous_queries, previous_query_states, previous_query, input_schema,final_utterance_states_h):
        previous_queries.append(previous_query)
        num_queries_to_keep = min(self.params.maximum_queries, len(previous_queries))
        previous_queries = previous_queries[-num_queries_to_keep:]
        query_token_embedder = lambda query_token:self.get_query_token_embedding(query_token, input_schema)
        previous_final_state, previous_outputs = self.query_encoder(previous_query, query_token_embedder, dropout_amount=self.dropout)

        assert len(previous_outputs) == len(previous_query)
        previous_query_states.append(previous_outputs)
        previous_query_states = previous_query_states[-num_queries_to_keep:]
        
      

        return previous_queries, previous_query_states,previous_final_state[1][-1]
    def train_step(self, interaction, max_generation_length, snippet_alignment_probability=1.,epoch=-1000):
        """ Trains the interaction-level model on a single interaction.

        Inputs:
            interaction (Interaction): The interaction to train on.
            learning_rate (float): Learning rate to use.
            snippet_keep_age (int): Age of oldest snippets to use.
            snippet_alignment_probability (float): The probability that a snippet will
                be used in constructing the gold sequence.
        """

        losses = []
        total_gold_tokens = 0

        input_hidden_states = []
        input_sequences = []


        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        input_schema = interaction.get_schema()
        schema_states = []
        previous_final_state=torch.cuda.FloatTensor(300).fill_(0)
      
        from_emb=[]
        final_utterance_states_h=[]
   
        previous_queries = []
        previous_query=[]
        previous_query_states = []
        now_dict=dict()
        now_dict['select']=[]
        now_dict['from']=[]
        now_dict['where']=[]
        sim_all_list_in=[]
        sim_all_tks_list_in =[]
        now_position=0
        for utterance_index, utterance in enumerate(interaction.gold_utterances()):
            if interaction.identifier in LIMITED_INTERACTIONS and utterance_index > LIMITED_INTERACTIONS[interaction.identifier]:
                break

            utterance_types=utterance.get_utterance_types()
            utterance_toks=utterance.get_utterance_toks()
            utterance_toks.append(['|'])
            input_sequence=[]
            input_sequence_types=[]
            idx_list=[]
            info_list=[]
            for tks,t_type in zip(utterance_toks,utterance_types):
                if tks==['|']:
                    break
                st_idx=len(idx_list)
                if t_type[0]!='te8r2ed':
                    input_sequence.append(t_type[0])
                    input_sequence_types.append('te8r2ed')
                for tk in tks:
                    input_sequence.append(tk)
                    input_sequence_types.append('te8r2ed')
                en_idx=len(input_sequence)
                idx_list.append([st_idx,en_idx])

       
            final_utterance_state, utterance_states, schema_states ,sim_all_list_cur,sim_all_tks_list_cur= self.get_bert_encoding(input_sequence,
                                                                                            input_schema,
                                                                                            discourse_state,idx_list,now_dict,
                                                                                            dropout=True,max_seq_length=350,now_position=now_position)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)
            sim_all_list_in.append(sim_all_list_cur)
            sim_all_tks_list_in.append(sim_all_tks_list_cur)
            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))
           
            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states, self.dropout)
            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence,_,sim_all_list,sim_all_tks_list = self._add_positional_embeddings(input_hidden_states, input_sequences,sim_all_list_in,sim_all_tks_list_in)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
            if self.params.use_query_attention and len(previous_query) > 0:
                previous_queries, previous_query_states,previous_final_state = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema,final_utterance_states_h)

            gold_match_id=[]
            gold_query = utterance.gold_query()
            if len(previous_query)>0:
                gold_match_id=cal_idx(previous_query,gold_query)
            else:
                gold_match_id=[-1]*len(gold_query)
          

            if len(gold_query) <= max_generation_length and len(previous_query) <= max_generation_length:
                prediction,now_dict,_ = self.predict_turn(final_utterance_state,
                flat_sequence,
                sim_all_list,sim_all_tks_list,
                                               utterance_states,
                                               schema_states,
                                               previous_queries,
                                               previous_query_states,
                                               previous_final_state,
                                               gold_match_id,
                                               info_list,
                                               max_generation_length,
                                               gold_query=gold_query,
                                               snippets=snippets,
                                             input_sequence=input_sequence,
                                               input_schema=input_schema,
                                               feed_gold_tokens=True,
                                               training=True,epoch=epoch,from_emb=from_emb,bf_dict=now_dict)
                loss = prediction[1]
                total_gold_tokens += len(gold_query)
                losses.append(loss)
                previous_query=gold_query[:-1]
                final_utterance_states_h.append(final_utterance_state[1][0])
            else:
                # Break if previous decoder snippet encoding -- because the previous
                # sequence was too long to run the decoder.
                if self.params.previous_decoder_snippet_encoding:
                    break
                continue

            torch.cuda.empty_cache()
            now_position+=1
        if losses:
            average_loss = torch.sum(torch.stack(losses)) / total_gold_tokens

            # Renormalize so the effect is normalized by the batch size.
            normalized_loss = average_loss
            if self.params.reweight_batch:
                normalized_loss = len(losses) * average_loss / float(self.params.batch_size)

            normalized_loss.backward()
            self.trainer.step()
            self.bert_trainer.step()
            self.zero_grad()
            loss_scalar = normalized_loss.item()
        else:
            loss_scalar = 0.
        return loss_scalar

    def predict_with_predicted_queries(self, interaction, max_generation_length,from_data, syntax_restrict=True,all_pre=None,allcountries=None):
        """ Predicts an interaction, using the predicted queries to get snippets."""
        # assert self.params.discourse_level_lstm
        
        syntax_restrict=False

        predictions = []

        input_hidden_states = []
        input_sequences = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()
        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []
        previous_queries = []
        previous_query_states = []
        previous_query=[]
        now_dict=dict()
        now_dict['select']=[]
        now_dict['from']=[]
        now_dict['where']=[]
        final_utterance_states_h = []
        sim_all_list_in=[]
        sim_all_tks_list_in=[]
        previous_final_state=torch.cuda.FloatTensor(300).fill_(0)     
        now_position=0
        interaction.start_interaction()
        while not interaction.done():
            utterance = interaction.next_utterance()
            available_snippets = utterance.snippets()
            utterance_types=utterance.get_utterance_types()
            utterance_toks=utterance.get_utterance_toks()
            input_sequence=[]
            input_sequence_types=[]
            idx_list=[]
            _tk_id=0
            utterance_toks.append(['|'])

            for tks,t_type in zip(utterance_toks,utterance_types):
                if tks==['|']:
                    break
                st_idx=len(idx_list)
                if t_type[0]!='te8r2ed':
                    input_sequence.append(t_type[0])
                    input_sequence_types.append('te8r2ed')
                for tk in tks:
                    input_sequence.append(tk)
                    input_sequence_types.append('te8r2ed')
                en_idx=len(input_sequence)
                idx_list.append([st_idx,en_idx])
                _tk_id+=1
    
                
            final_utterance_state, utterance_states, schema_states,sim_all_list_cur,sim_all_tks_list_cur=self.get_bert_encoding(input_sequence,input_schema,discourse_state,idx_list,now_dict,dropout=True,max_seq_length=512)

            sim_all_list_in.append(sim_all_list_cur)
            sim_all_tks_list_in.append(sim_all_tks_list_cur)
            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)
            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states)
            utterance_states, flat_sequence,_,sim_all_list,sim_all_tks_list  = self._add_positional_embeddings(input_hidden_states, input_sequences,sim_all_list_in,sim_all_tks_list_in)
      

            snippets = None
            if self.params.use_snippets:
                snippets = self._encode_snippets(previous_query, available_snippets, input_schema)

            if self.params.use_query_attention and  len(previous_query) > 0:
                previous_queries, previous_query_states,previous_final_state = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema,final_utterance_states_h)

            assert len(sim_all_tks_list)==len(sim_all_list) 
            assert len(sim_all_list)==0
            from_emb=[]
            all_pre=[1]
            gold_match_id=[]
            results,now_dict,_ = self.predict_turn(final_utterance_state,
             flat_sequence,
                sim_all_list,sim_all_tks_list,
                                        utterance_states,
                                        schema_states,
                                        previous_queries,
                                        previous_query_states,
                                        previous_final_state,
                                        gold_match_id,
                                        [],
                                        max_generation_length,
                                         input_sequence=input_sequence,
                                         input_sequences=input_sequences,
                                        input_schema=input_schema,
                                        snippets=snippets,from_emb=from_emb,cmp_pre=all_pre[0],bf_dict=now_dict)
            if len(all_pre)!=1:
                all_pre=all_pre[1:]

            predicted_sequence = results[0]
            predictions.append(results)
            
            # Update things necessary for using predicted queries
            anonymized_sequence = utterance.remove_snippets(predicted_sequence)
            if EOS_TOK in anonymized_sequence:
                anonymized_sequence = anonymized_sequence[:-1] # Remove _EOS
            else:
                anonymized_sequence = ['select', '*', 'from', 't1']
            previous_query=predicted_sequence[:-1]
            if not syntax_restrict:
                utterance.set_predicted_query(interaction.remove_snippets(predicted_sequence))
                if input_schema:
                    # on SParC
                    interaction.add_utterance(utterance, anonymized_sequence, previous_snippets=utterance.snippets(), simple=True)
                else:
                    # on ATIS
                    interaction.add_utterance(utterance, anonymized_sequence, previous_snippets=utterance.snippets(), simple=False)
            else:
                utterance.set_predicted_query(utterance.previous_query())
                interaction.add_utterance(utterance, utterance.previous_query(), previous_snippets=utterance.snippets())
            now_position+=1
            final_utterance_states_h.append(final_utterance_state[1][0])

        return predictions,all_pre,input_schema


    def predict_with_gold_queries(self, interaction, max_generation_length, feed_gold_query=False,all_pre=None):
        """ Predicts SQL queries for an interaction.

        Inputs:
            interaction (Interaction): Interaction to predict for.
            feed_gold_query (bool): Whether or not to feed the gold token to the
                generation step.
        """
        feed_gold_query=False
        predictions = []
        input_hidden_states = []
        input_sequences = []
        previous_final_state=torch.cuda.FloatTensor(300).fill_(0)
        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []
        previous_queries = []
        previous_query_states = []
        previous_query=[]
        now_dict=dict()
        now_dict['select']=[]
        now_dict['from']=[]
        now_dict['where']=[]
     


        is_correct=[]
        sim_all_list_in=[]
        sim_all_tks_list_in =[]
        
        now_position=0
        final_utterance_states_h=[]
        for utterance in interaction.gold_utterances():
            utterance_types=utterance.get_utterance_types()
            utterance_toks=utterance.get_utterance_toks()
            utterance_toks.append(['|'])
            input_sequence=[]
            input_sequence_types=[]
            idx_list=[]
            info_list=[]
            for tks,t_type in zip(utterance_toks,utterance_types):
                if tks==['|']:
                    break
                st_idx=len(idx_list)
                if t_type[0]!='te8r2ed':
                    input_sequence.append(t_type[0])
                    input_sequence_types.append('te8r2ed')
                for tk in tks:
                    input_sequence.append(tk)
                    input_sequence_types.append('te8r2ed')
                en_idx=len(input_sequence)
                idx_list.append([st_idx,en_idx])
            from_emb=[]
            final_utterance_state, utterance_states, schema_states,sim_all_list_cur,sim_all_tks_list_cur=self.get_bert_encoding(input_sequence,input_schema,discourse_state,idx_list,now_dict,dropout=True,max_seq_length=350,now_position=now_position)
            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            sim_all_list_in.append(sim_all_list_cur)
            sim_all_tks_list_in.append(sim_all_tks_list_cur)
  
            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))

            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states, self.dropout)
            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence,_,sim_all_list,sim_all_tks_list = self._add_positional_embeddings(input_hidden_states, input_sequences,sim_all_list_in,sim_all_tks_list_in)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
           
            if self.params.use_query_attention and len(previous_query) > 0:
                previous_queries, previous_query_states,previous_final_state = self.get_previous_queries(previous_queries, previous_query_states, previous_query, input_schema,final_utterance_states_h)
            gold_query=utterance.gold_query()
            gold_match_id=[]
            if len(previous_query)>0:
                gold_match_id=cal_idx(previous_query,gold_query)
            else:
                gold_match_id=[-1]*len(gold_query)
            assert len(sim_all_tks_list)==len(sim_all_list) 
            assert len(sim_all_list)==0
        
            prediction,now_dict,is_corr = self.predict_turn(final_utterance_state,
                                        flat_sequence,
                                        sim_all_list,sim_all_tks_list,
                                           utterance_states,
                                           schema_states,
                                           previous_queries,
                                           previous_query_states,
                                           previous_final_state,
                                           gold_match_id,
                                           info_list,
                                           max_generation_length,

                                           gold_query=gold_query,
                                           snippets=snippets,
                                           input_sequence=input_sequence,
                                           input_schema=input_schema,
                                           feed_gold_tokens=True,from_emb=from_emb,bf_dict=now_dict)
            predictions.append(prediction)
            previous_query=gold_query[:-1]
         
            now_position +=1  
            is_correct.append(is_corr) 
            final_utterance_states_h.append(final_utterance_state[1][0])

      

        return predictions,all_pre,input_schema,is_correct

    def concat(self,sql1,pare):
        if sql1.count('from')!=1 :
            return sql1
        add_from=pare
        flagf=0
        for x in sql1:
            if x=='from':
                flagf=1
            if flagf==1 and x not in self.voc:
                add_from.append(x)
            if x !='from' and x in self.voc:
                flagf=0
        add_from=list(set(add_from))
        nsql=[]
        flagf=0
        for x in sql1:
            if x=='from':
                nsql.append('from')
                nsql.extend(add_from)
                flagf=1
       
            if x !='from' and x in self.voc:
                flagf=0
            if flagf==0:
                nsql.append(x)    
        return nsql




def cal_idx(previous_sql,gold_sql):
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


