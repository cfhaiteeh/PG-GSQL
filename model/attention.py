"""Contains classes for computing and keeping track of attention distributions.
"""
from collections import namedtuple

import torch
import torch.nn.functional as F
from . import torch_utils

class AttentionResult(namedtuple('AttentionResult',
                                 ('scores',
                                  'distribution',
                                  'vector'))):
    """Stores the result of an attention calculation."""
    __slots__ = ()


class Attention(torch.nn.Module):
    """Attention mechanism class. Stores parameters for and computes attention.

    Attributes:
       transform_query (bool): Whether or not to transform the query being
           passed in with a weight transformation before computing attentino.
       transform_key (bool): Whether or not to transform the key being
           passed in with a weight transformation before computing attentino.
       transform_value (bool): Whether or not to transform the value being
           passed in with a weight transformation before computing attentino.
       key_size (int): The size of the key vectors.
       value_size (int): The size of the value vectors.
           the query or key.
       query_weights (dy.Parameters): Weights for transforming the query.
       key_weights (dy.Parameters): Weights for transforming the key.
       value_weights (dy.Parameters): Weights for transforming the value.
    """
    def __init__(self, query_size, key_size, value_size,name='weights-attention-q'):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        self.query_weights = torch_utils.add_params((query_size, self.key_size), name)

    def transform_arguments(self, query, keys, values):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
        assert len(keys) == len(values)

        all_keys = torch.stack(keys, dim=1)
        all_values = torch.stack(values, dim=1)

        assert all_keys.size()[0] == self.key_size, "Expected key size of " + str(self.key_size) + " but got " + str(all_keys.size()[0])
        assert all_values.size()[0] == self.value_size

        query = torch_utils.linear_layer(query, self.query_weights)

        return query, all_keys, all_values

    def Calquery(self, query):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
     

       

        query = torch_utils.linear_layer(query, self.query_weights)

        return query

    def forward(self, query, keys, values=None):
        if not values:
            values = keys

        query_t, keys_t, values_t = self.transform_arguments(query, keys, values)

        scores = torch.t(torch.mm(query_t,keys_t))      
        distribution = F.softmax(scores, dim=0)          
        context_vector = torch.mm(values_t, distribution).squeeze()  
        query_t.cpu()
        keys_t.cpu()
        values_t.cpu()
        return AttentionResult(scores, distribution, context_vector)
    


class AttentionWeight(torch.nn.Module):
    """Attention mechanism class. Stores parameters for and computes attention.

    Attributes:
       transform_query (bool): Whether or not to transform the query being
           passed in with a weight transformation before computing attentino.
       transform_key (bool): Whether or not to transform the key being
           passed in with a weight transformation before computing attentino.
       transform_value (bool): Whether or not to transform the value being
           passed in with a weight transformation before computing attentino.
       key_size (int): The size of the key vectors.
       value_size (int): The size of the value vectors.
           the query or key.
       query_weights (dy.Parameters): Weights for transforming the query.
       key_weights (dy.Parameters): Weights for transforming the key.
       value_weights (dy.Parameters): Weights for transforming the value.
    """
    def __init__(self, query_size, key_size, value_size,name='weights-attention-q'):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        self.query_weights = torch_utils.add_params((query_size, self.key_size), name)

    def transform_arguments(self, query, keys, values):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
        assert len(keys) == len(values)

        all_keys = torch.stack(keys, dim=1)
        all_values = torch.stack(values, dim=1)

        assert all_keys.size()[0] == self.key_size, "Expected key size of " + str(self.key_size) + " but got " + str(all_keys.size()[0])
        assert all_values.size()[0] == self.value_size

        query = torch_utils.linear_layer(query, self.query_weights)

        return query, all_keys, all_values

    def Calquery(self, query):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
     

       

        query = torch_utils.linear_layer(query, self.query_weights)

        return query

    def forward(self, query, keys, values=None):
        if not values:
            values = keys

        query_t, keys_t, _ = self.transform_arguments(query, keys, values)

        scores = torch.t(torch.mm(query_t,keys_t))        
        distribution = F.softmax(scores, dim=0)        
        return distribution
    


class AttentionAddMode(torch.nn.Module):
    """Attention mechanism class. Stores parameters for and computes attention.

    Attributes:
       transform_query (bool): Whether or not to transform the query being
           passed in with a weight transformation before computing attentino.
       transform_key (bool): Whether or not to transform the key being
           passed in with a weight transformation before computing attentino.
       transform_value (bool): Whether or not to transform the value being
           passed in with a weight transformation before computing attentino.
       key_size (int): The size of the key vectors.
       value_size (int): The size of the value vectors.
           the query or key.
       query_weights (dy.Parameters): Weights for transforming the query.
       key_weights (dy.Parameters): Weights for transforming the key.
       value_weights (dy.Parameters): Weights for transforming the value.
    """
    def __init__(self, query_size, key_size, value_size,name='weights-attention-q'):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        self.query_weights = torch_utils.add_params((query_size, key_size), name+'query')
        self.keys_weights = torch_utils.add_params((key_size, key_size), name+'keys')
        self.V_weights = torch_utils.add_params((key_size, 1), name+'V')



    def transform_arguments(self, query, keys, values):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
        assert len(keys) == len(values)

        all_keys = torch.stack(keys, dim=1)
        all_values = torch.stack(values, dim=1)

        assert all_keys.size()[0] == self.key_size, "Expected key size of " + str(self.key_size) + " but got " + str(all_keys.size()[0])
        assert all_values.size()[0] == self.value_size

        query = torch_utils.linear_layer(query, self.query_weights)
        

        return query, all_keys, all_values

    def Calquery(self, query):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
     

       

        query = torch_utils.linear_layer(query, self.query_weights)

        return query

    def forward(self, query, keys, values=None):
        if not values:
            values = keys


        query = torch_utils.linear_layer(query, self.query_weights)
        all_keys = torch.stack(keys, dim=0)
        l,r=list(all_keys.size())
        query=query.expand(l,r).contiguous()
       
        keys=torch_utils.linear_layer(all_keys, self.keys_weights)
     
        att_features = query + keys 
        e = torch.tanh(att_features)

        scores = torch_utils.linear_layer(e,  self.V_weights)    

        scores = scores.view(-1, l) 
        distribution = F.softmax(scores, dim=1)        
        c_t = torch.mm(distribution, all_keys).squeeze() 
        return AttentionResult(scores, distribution, c_t)
    





class AttentionAddMode_LOCAL(torch.nn.Module):
    """Attention mechanism class. Stores parameters for and computes attention.

    Attributes:
       transform_query (bool): Whether or not to transform the query being
           passed in with a weight transformation before computing attentino.
       transform_key (bool): Whether or not to transform the key being
           passed in with a weight transformation before computing attentino.
       transform_value (bool): Whether or not to transform the value being
           passed in with a weight transformation before computing attentino.
       key_size (int): The size of the key vectors.
       value_size (int): The size of the value vectors.
           the query or key.
       query_weights (dy.Parameters): Weights for transforming the query.
       key_weights (dy.Parameters): Weights for transforming the key.
       value_weights (dy.Parameters): Weights for transforming the value.
    """
    def __init__(self, query_size, key_size, value_size,name='weights-attention-q'):
        super().__init__()
        self.key_size = key_size
        self.value_size = value_size

        self.query_weights = torch_utils.add_params((query_size, key_size), name+'query')
        self.keys_weights = torch_utils.add_params((key_size, key_size), name+'keys')
        self.V_weights = torch_utils.add_params((key_size, 1), name+'V')



    def transform_arguments(self, query, keys, values):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
        assert len(keys) == len(values)

        all_keys = torch.stack(keys, dim=1)
        all_values = torch.stack(values, dim=1)

        assert all_keys.size()[0] == self.key_size, "Expected key size of " + str(self.key_size) + " but got " + str(all_keys.size()[0])
        assert all_values.size()[0] == self.value_size

        query = torch_utils.linear_layer(query, self.query_weights)
        

        return query, all_keys, all_values

    def Calquery(self, query):
        """ Transforms the query/key/value inputs before attention calculations.

        Arguments:
            query (dy.Expression): Vector representing the query (e.g., hidden state.)
            keys (list of dy.Expression): List of vectors representing the key
                values.
            values (list of dy.Expression): List of vectors representing the values.

        Returns:
            triple of dy.Expression, where the first represents the (transformed)
                query, the second represents the (transformed and concatenated)
                keys, and the third represents the (transformed and concatenated)
                values.
        """
     

       

        query = torch_utils.linear_layer(query, self.query_weights)

        return query

    def forward(self, query, keys,id_list, values=None):
        if not values:
            values = keys


        query = torch_utils.linear_layer(query, self.query_weights)
        all_keys = torch.stack(keys, dim=0)

        l,r=list(all_keys.size())
        query=query.expand(l,r).contiguous()
       
        keys=torch_utils.linear_layer(all_keys, self.keys_weights)
     
        att_features = query + keys 
        e = torch.tanh(att_features)

        scores = torch_utils.linear_layer(e,  self.V_weights)     
        scores = scores.view(-1, l) 
        new_scores=[]
        new_all_keys=[]
        for id_1 in id_list:
            new_scores.append(scores[0][id_1])
            new_all_keys.append(keys[id_1])
        new_all_keys = torch.stack(new_all_keys, dim=0)
        new_scores=torch.stack(new_scores,dim=0)
        new_scores=new_scores.unsqueeze(0)
        distribution = F.softmax(new_scores, dim=1)    
        c_t = torch.mm(distribution, new_all_keys).squeeze() 
        return AttentionResult(scores, distribution, c_t)
    