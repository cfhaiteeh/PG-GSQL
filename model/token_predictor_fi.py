"""Predicts a token."""

from collections import namedtuple

import torch
import torch.nn.functional as F
from . import torch_utils
import copy
from .attention import Attention, AttentionResult,AttentionWeight,AttentionAddMode,AttentionAddMode_LOCAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PredictionInput(namedtuple('PredictionInput',
                                 ('decoder_state',
                                  'input_hidden_states',
                                  'snippets',
                                  'input_sequence'))):
    """ Inputs to the token predictor. """
    __slots__ = ()

class PredictionInputWithSchema(namedtuple('PredictionInputWithSchema',
                                 ('decoder_state',
                                  'input_hidden_states',
                                  'schema_states',
                                  'previous_queries',
                                  'previous_query_states',
                                  'snippets',
                                  'input_sequence',
                                  'input_schema','tks','flag','schema_v'))):
    """ Inputs to the token predictor. """
    __slots__ = ()
class TokenPrediction(namedtuple('TokenPrediction',
                                 ('scores',
                                  'aligned_tokens',
                                  'utterance_attention_results',
                                  'decoder_state','aug','query_attention_weight','switch','question_attention_weight','gen_switch'
                                   ))):

    """A token prediction.

    Attributes:
        scores (dy.Expression): Scores for each possible output token.
        aligned_tokens (list of str): The output tokens, aligned with the scores.
        attention_results (AttentionResult): The result of attending on the input
            sequence.
    """
    __slots__ = ()

def score_snippets(snippets, scorer):
    """ Scores snippets given a scorer.

    Inputs:
        snippets (list of Snippet): The snippets to score.
        scorer (dy.Expression): Dynet vector against which to score  the snippets.

    Returns:
        dy.Expression, list of str, where the first is the scores and the second
            is the names of the snippets that were scored.
    """
    snippet_expressions = [snippet.embedding for snippet in snippets]
    all_snippet_embeddings = torch.stack(snippet_expressions, dim=1)

    scores = torch.t(torch.mm(torch.t(scorer), all_snippet_embeddings))

    if scores.size()[0] != len(snippets):
        raise ValueError("Got " + str(scores.size()[0]) + " scores for " + str(len(snippets)) + " snippets")

    return scores, [snippet.name for snippet in snippets]

def score_schema_tokens(input_schema, schema_states, scorer):
    scores = torch.t(torch.mm(torch.t(scorer), schema_states))   # num_tokens x 1
    return scores, input_schema.column_names_surface_form
def score_previous_tokens(previous_tokens_now, previous_query_state_now, scorer):
    scores = torch.t(torch.mm(torch.t(scorer), previous_query_state_now))   # num_tokens x 1
    return scores, previous_tokens_now

class TokenPredictor(torch.nn.Module):
    """ Predicts a token given a (decoder) state.

    Attributes:
        vocabulary (Vocabulary): A vocabulary object for the output.
        attention_module (Attention): An attention module.
        state_transformation_weights (dy.Parameters): Transforms the input state
            before predicting a token.
        vocabulary_weights (dy.Parameters): Final layer weights.
        vocabulary_biases (dy.Parameters): Final layer biases.
    """

    def __init__(self, params, vocabulary, attention_key_size):
        super().__init__()
        self.params = params
        self.vocabulary = vocabulary
        self.attention_module = AttentionAddMode(params.decoder_state_size, attention_key_size-50, attention_key_size-50,name='attention_module_fi')
        schema_attention_key_size=300
        self.col_attention_module = AttentionAddMode_LOCAL(300, schema_attention_key_size, schema_attention_key_size,name='col_attention_module_fi')
        self.start_schema_attention_vector = torch_utils.add_params((300,), "start_schema_attention_vector_fi")
        if params.use_query_attention:
            if self.params.discourse_level_lstm:
                self.in_size=300+300+300+schema_attention_key_size
            else:
                self.in_size=300+300+300+schema_attention_key_size
        else:
            if self.params.discourse_level_lstm:
                self.in_size=300+300+schema_attention_key_size
            else:
                self.in_size=300+300+schema_attention_key_size
        self.in_size-=300
        self.state_transform_weights = torch_utils.add_params((self.in_size, 300), "weights-state-transform_fi")
        self.vocabulary_weights = torch_utils.add_params((300,len(vocabulary)), "weights-vocabulary_fi")
        self.vocabulary_biases = torch_utils.add_params(tuple([len(vocabulary)]), "biases-vocabulary_fi")

    def _get_intermediate_state(self, state, dropout_amount=0.):
        intermediate_state = (torch_utils.linear_layer(state, self.state_transform_weights))

        return F.dropout(intermediate_state, dropout_amount)

    def _score_vocabulary_tokens(self, state):
        scores = torch.t(torch_utils.linear_layer(state, self.vocabulary_weights, self.vocabulary_biases))
        if scores.size()[0] != len(self.vocabulary.inorder_tokens):
            raise ValueError("Got " + str(scores.size()[0]) + " scores for " + str(len(self.vocabulary.inorder_tokens)) + " vocabulary items")

        return scores, self.vocabulary.inorder_tokens

    def forward(self, tabok,tbidx, tabs,allTable,IdxCol,prediction_input, dropout_amount=0.):
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states

        attention_results = self.attention_module(decoder_state, input_hidden_states)

        state_and_attn = torch.cat([decoder_state, attention_results.vector], dim=0)

        intermediate_state = self._get_intermediate_state(state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(intermediate_state)

        return TokenPrediction(vocab_scores, vocab_tokens, attention_results, decoder_state)


class SchemaTokenPredictor(TokenPredictor):
    """ Token predictor that also predicts snippets.

    Attributes:
        snippet_weights (dy.Parameter): Weights for scoring snippets against some
            state.
    """

    def __init__(self, params, vocabulary, utterance_attention_key_size, schema_attention_key_size, snippet_size):
        TokenPredictor.__init__(self, params, vocabulary, utterance_attention_key_size)
        if params.use_snippets:
            if snippet_size <= 0:
                raise ValueError("Snippet size must be greater than zero; was " + str(snippet_size))
            self.snippet_weights = torch_utils.add_params((params.decoder_state_size, snippet_size), "weights-snippet")

        schema_attention_key_size=300
        self.schema_token_weights = torch_utils.add_params((300, schema_attention_key_size), "weights-schema-token-fi")
       

    def _get_schema_token_scorer(self, state):
        scorer = torch.t(torch_utils.linear_layer(state, self.schema_token_weights))
        return scorer

    def _get_table_token_scorer(self, state):
        scorer = torch.t(torch_utils.linear_layer(state, self.tables_token_weights))
        return scorer


    def forward(self,tabok,tbidx, tabs,allTable,IdxCol, prediction_input,tb_col,previous_query_now,previous_query_state_now,decoder_input,schem_map,output_embedder,query_attention_results_dis, dropout_amount=0.):

        id_list=[]
        for cn in (prediction_input.tks):
                cur_id=schem_map[cn]
                id_list.append(cur_id)

        
            
        decoder_state = prediction_input.decoder_state
        input_hidden_states = prediction_input.input_hidden_states
        input_schema = prediction_input.input_schema
        schema_states = prediction_input.schema_states
        schema_v = prediction_input.schema_v
     
        utterance_attention_results = self.attention_module(decoder_state, input_hidden_states)
        if len(id_list)!=0:
            schema_attention_results=self.col_attention_module(decoder_state,schema_v,id_list).vector
        else:
            schema_attention_results = self.start_schema_attention_vector
            schema_attention_results = AttentionResult(None, None, schema_attention_results).vector
        state_and_attn = torch.cat([decoder_state, utterance_attention_results.vector.cuda(),schema_attention_results], dim=0)
        c_k=torch.cat([utterance_attention_results.vector.cuda(),schema_attention_results],dim=0)
        switch=None
        query_attention_weight=None
        gen_switch=None
        intermediate_state = self._get_intermediate_state(state_and_attn, dropout_amount=dropout_amount)
        vocab_scores, vocab_tokens = self._score_vocabulary_tokens(intermediate_state)
        final_scores = [_ for _ in vocab_scores]
        aligned_tokens = []
        aligned_tokens.extend(vocab_tokens)
        if len(schema_states)!=0:
            schema_states = torch.stack(schema_states, dim=1)
            schema_scores, _ = score_schema_tokens(input_schema, schema_states, self._get_schema_token_scorer(intermediate_state))
            final_scores.extend(schema_scores)
            aligned_tokens.extend(prediction_input.tks)
                
        final_scores=torch.stack(final_scores, dim=0)
        final_scores = final_scores.squeeze()

        return TokenPrediction(final_scores, aligned_tokens, utterance_attention_results, decoder_state,c_k,query_attention_weight,switch,utterance_attention_results.distribution,gen_switch)






def construct_token_predictor_fi(params,
                              vocabulary,
                              utterance_attention_key_size,
                              schema_attention_key_size,
                              snippet_size,
                              anonymizer=None):
    """ Constructs a token predictor given the parameters.

    Inputs:
        parameter_collection (dy.ParameterCollection): Contains the parameters.
        params (dictionary): Contains the command line parameters/hyperparameters.
        vocabulary (Vocabulary): Vocabulary object for output generation.
        attention_key_size (int): The size of the attention keys.
        anonymizer (Anonymizer): An anonymization object.
    """


    if not anonymizer and not params.previous_decoder_snippet_encoding:
        print('using SchemaTokenPredictor')
        return SchemaTokenPredictor(params, vocabulary, utterance_attention_key_size, schema_attention_key_size, snippet_size)
    else:
        print('Unknown token_predictor')
        exit()
