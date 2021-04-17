""" Contains the class for an interaction in ATIS. """

from . import anonymization as anon
from . import sql_util
from .snippets import expand_snippets
from .utterance import Utterance, OUTPUT_KEY, ANON_INPUT_KEY

import torch


def getl_r(cl,ty,column_names_embedder_input,column_type):
    ti=0
    se=[]
    zi=-1
    for tb,tbty in zip(column_names_embedder_input,column_type):
        if tbty=='table':
            zi+=1
            if cl+' .' in tb:
                idx=tb.index(cl+' .')
                tl=tb.split()
                l=0
                zb=-1
                yb=-1
                for oi,zz in enumerate(tl):
                    if l>=idx:
                        if(zb==-1):
                            zb=oi
                        if zz=='.':
                            return zb,yb,zi
                        yb=oi
                        se.append(zz)
                    else:
                        l+=len(zz)
                        l+=1
            ti+=1

class Schema:
    def __init__(self, table_schema, simple=False):
        if simple:
            self.helper1(table_schema)

    def helper1(self, table_schema):
      
        self.foreign_keys=table_schema['foreign_keys']
        self.column_names = table_schema['column_names']
        self.column_names_original=table_schema['column_names_original']
        self.table_names = table_schema['table_names']
        self.table_names_original=table_schema['table_names_original']
        self.column_value_types=table_schema['column_types']
        self.primary_keys=table_schema['primary_keys']
        
        self.ys_map_o=[x[1].lower() for x in self.column_names_original]
        self.ys_map=[x[1].lower() for x in self.column_names]
        
        column_keep_index = []
        

      

        self.column_names_embedder_input = []
        self.link_table=[]
        self.column_type=[]
        self.usedCID=[]
        self.usedTID=[]
        column_keep_index = []
        self.column_names_embedder_input_to_id = {}

      
        for i, ((table_id, column_name),cvt) in enumerate(zip(self.column_names,self.column_value_types)):

      
            column_name_embedder_input = str(column_name.lower())

            column_name_embedder_input = 'column '+str(column_name.lower())
          
            if column_name_embedder_input not in self.column_names_embedder_input_to_id:

                self.column_names_embedder_input.append(column_name_embedder_input)
              
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1
                self.column_type.append('column')
                self.usedCID.append(i)
      
                if table_id==-1:
                    self.link_table.append('te8r2ed')
                else:
                    self.link_table.append(self.table_names[table_id])
                column_keep_index.append(i)
        self.table_link=dict()
        column_keep_index_2 = []
  
           

        for i, table_name in enumerate(self.table_names):
            column_name_embedder_input = str(table_name.lower())
            column_name_embedder_input = 'table '+str(table_name.lower())
            self.table_link[column_name_embedder_input]=[]
          
            if column_name_embedder_input not in self.column_names_embedder_input_to_id:
                for ic, ((table_id, column_name),cvt) in enumerate(zip(self.column_names,self.column_value_types)):
                    column_name_l = column_name.lower()
                    if table_id==i:
                       
                        column_name_embedder_input=('column '+column_name_l)+' . '+column_name_embedder_input
                
                

                
                self.column_names_embedder_input.append(column_name_embedder_input)
             
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1
                self.column_type.append('table')
                self.usedTID.append(i)
                self.link_table.append('te8r2ed')
                self.table_link[column_name_embedder_input]=[]
                column_keep_index_2.append(i)
       
        self.indexofcl={}
        for cl,ty in zip(self.column_names_embedder_input,self.column_type):
            if ty=='column' and cl !='column *':
                l,r,z=getl_r(cl,ty,self.column_names_embedder_input,self.column_type)
                self.indexofcl[cl]=[l,r,z]
        self.column_names_surface_form = []
        self.column_names_surface_form_to_id = {}
        self.column_value_type_to_id={}
        for i, ((table_id, column_name),cvt) in enumerate(zip(self.column_names_original,self.column_value_types)):
            column_name_surface_form = column_name
          
            column_name_surface_form =column_name_surface_form.lower()

            if i in column_keep_index:
                self.column_names_surface_form.append(column_name_surface_form)
                if column_name_surface_form not in self.column_names_surface_form_to_id:
                    self.column_value_type_to_id[column_name_surface_form]=cvt
                    self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
        

        self.split_len=len(self.column_names_surface_form)
        for i, table_name in enumerate(self.table_names_original):
          
            column_name_surface_form =table_name.lower()

            if i in column_keep_index_2:
                self.column_names_surface_form.append(column_name_surface_form)
                if column_name_surface_form not in self.column_names_surface_form_to_id:

                    self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
              


       
        max_id_2 = max(v for k,v in self.column_names_embedder_input_to_id.items())
      

        assert (len(self.column_names_surface_form) - 1) == max_id_2 

        self.num_col = len(self.column_names_surface_form)
       

    def utterance_change(self,used_tbs_ids_list):
  
   


      
        self.column_names_embedder_input = []
        self.link_table=[]
        self.column_type=[]
        self.usedCID=[]
        self.usedTID=[]
        column_keep_index = []
        self.column_names_embedder_input_to_id = {}


         
        for i, (table_id, column_name) in enumerate(self.column_names):
            if table_id not in used_tbs_ids_list and table_id!=-1:
                continue
            column_name_embedder_input = str(column_name.lower())

            column_name_embedder_input = 'column '+str(column_name.lower())
          
            if column_name_embedder_input not in self.column_names_embedder_input_to_id:
                self.column_names_embedder_input.append(column_name_embedder_input)
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1
                self.column_type.append('column')
                self.usedCID.append(i)
        
                if table_id==-1:
                    self.link_table.append('te8r2ed')
                else:
                    self.link_table.append(self.table_names[table_id])
                column_keep_index.append(i)
        column_keep_index_2 = []
        for i, table_name in enumerate(self.table_names):
            if i not in used_tbs_ids_list:
                continue
            column_name_embedder_input = str(table_name.lower())

            column_name_embedder_input = 'table '+str(table_name.lower())
            self.table_link[column_name_embedder_input]=[]
            if column_name_embedder_input not in self.column_names_embedder_input_to_id:
                
                for ic, (table_id, column_name) in enumerate(self.column_names):
                    column_name_l = column_name.lower()
                    if table_id==i:
                        column_name_embedder_input=('column '+column_name_l)+' . '+column_name_embedder_input
                
                


                self.column_names_embedder_input.append(column_name_embedder_input)
               
                self.column_names_embedder_input_to_id[column_name_embedder_input] = len(self.column_names_embedder_input) - 1
                self.column_type.append('table')
                self.usedTID.append(i)
                self.link_table.append('te8r2ed')
                self.table_link[column_name_embedder_input]=[]
                column_keep_index_2.append(i)
  
        self.column_names_surface_form = []
        self.column_names_surface_form_to_id = {}
        for i, (table_id, column_name) in enumerate(self.column_names_original):
            column_name_surface_form = column_name
          
            column_name_surface_form =column_name_surface_form.lower()

            if i in column_keep_index:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
        

        self.split_len=len(self.column_names_surface_form)
        for i, table_name in enumerate(self.table_names_original):
         
            column_name_surface_form =table_name.lower()

            if i in column_keep_index_2:
                self.column_names_surface_form.append(column_name_surface_form)
                self.column_names_surface_form_to_id[column_name_surface_form] = len(self.column_names_surface_form) - 1
              




        
        max_id_1 = max(v for k,v in self.column_names_surface_form_to_id.items())
        max_id_2 = max(v for k,v in self.column_names_embedder_input_to_id.items())
  
        assert (len(self.column_names_surface_form) - 1) == max_id_2 == max_id_1
        assert len(self.column_names_embedder_input)==len(self.column_names_surface_form)
        self.num_col = len(self.column_names_surface_form)
   
    def __len__(self):
        return self.num_col

    def in_vocabulary(self, column_name, surface_form=False):
      
        if surface_form:
            return column_name in self.column_names_surface_form_to_id
        else:
            return column_name in self.column_names_embedder_input_to_id
    
    def cal_table_link(self,link_list,column_name_token_embedder=None):
        if len(link_list)==0:
            return torch.cuda.FloatTensor(300).fill_(0)
        ans=[]
        for x in link_list:
    
            column_embeddings=[column_name_token_embedder.get(token) for token in x.split()]
            column_embeddings=torch.stack(column_embeddings, dim=0)
            column_embeddings=torch.mean(column_embeddings, dim=0)
            ans.append(column_embeddings)
        ans_emb=torch.stack(ans, dim=0)
        ans_emb=torch.mean(ans_emb, dim=0)

        return ans_emb


    def cal_table_link_bert(self,link_list,column_name_token_embedder=None):
        if len(link_list)==0:
            return torch.cuda.FloatTensor(300).fill_(0)
        ans=[]
  
 
        for x in link_list:

            idx=self.column_names_surface_form.index(x)

            ans.append(column_name_token_embedder[idx])

        ans_emb=torch.stack(ans, dim=0)
        ans_emb=torch.mean(ans_emb, dim=0)

        return ans_emb

    def set_table_link_emb(self,emb):
        self.table_link_emb=emb

    def column_name_embedder_bow(self, column_name, surface_form=False, column_name_token_embedder=None):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_surface_form_to_id[column_name]
            column_name_embedder_input = self.column_names_embedder_input[column_name_id]
    

        else:
 
            column_name_embedder_input = column_name
   

        column_name_embeddings = [column_name_token_embedder(token) for token in column_name_embedder_input.split()]
        column_name_embeddings = torch.stack(column_name_embeddings, dim=0)
        return torch.mean(column_name_embeddings, dim=0)

    def column_name_embedder_bow_with_type(self, column_name,column_type,link_table, surface_form=False, column_name_token_embedder=None):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            column_name_id = self.column_names_surface_form_to_id[column_name]
            column_name_embedder_input = self.column_names_embedder_input[column_name_id]
   
        else:
            column_name_embedder_input = column_name
     
        column_type_embeddings=[column_name_token_embedder.get(token) for token in column_type.split()]
        column_type_embeddings=torch.stack(column_type_embeddings, dim=0)
        column_type_embeddings=torch.mean(column_type_embeddings, dim=0)

        
        column_name_embeddings = [column_name_token_embedder.get(token) for token in column_name_embedder_input.split()]
        column_name_embeddings = torch.stack(column_name_embeddings, dim=0)
       
        column_name_embeddings=[torch.mean(column_name_embeddings, dim=0),column_type_embeddings]
              
       
        column_name_embeddings = torch.stack(column_name_embeddings, dim=0)
        return torch.mean(column_name_embeddings, dim=0)



    def set_from_name_embeddings(self, column_name_embeddings):
        self.from_name_embeddings = column_name_embeddings
    
    def set_column_name_embeddings(self, column_name_embeddings):
        self.column_name_embeddings = column_name_embeddings
        assert len(self.column_name_embeddings) == self.num_col
    def from_embedder(self, column_name, surface_form=False):
        assert self.in_vocabulary(column_name, surface_form)
        if surface_form:
            if column_name not in self.foreign_graph_in_surface_form:
                return self.column_name_embedder(column_name,surface_form=True)
            column_name_id = self.foreign_graph_in_id_surface_form[column_name]
           
        else:
            if column_name not in self.foreign_graph_in:
                return self.column_name_embedder(column_name,surface_form=False)
            column_name_id = self.foreign_graph_in_id[column_name]

        return self.from_name_embeddings[column_name_id]
    def column_name_embedder(self, column_name, surface_form=False):
       
        if surface_form:
            if column_name in  self.column_names_surface_form_to_id:
                
                column_name_id = self.column_names_surface_form_to_id[column_name]
            else:
                ys1=self.ys_map_o.index(column_name)
                cn=self.ys_map[ys1]
               
                column_name_id = self.column_names_embedder_input_to_id['column '+cn]

        else:
            column_name_id = self.column_names_embedder_input_to_id[column_name]

        return self.column_name_embeddings[column_name_id]


    def foreign_graph(self):
        self.foreign_graph_in=[]
        self.foreign_graph_in_id=dict()
        self.node_type=[]

        self.foreign_graph_in_surface_form=[]
        self.foreign_graph_in_id_surface_form=dict()
        for i, table_name in enumerate(self.table_names_original):
            column_name_embedder_input = table_name.lower()
            self.foreign_graph_in_surface_form.append(column_name_embedder_input)
            self.node_type.append('table')
            self.foreign_graph_in_id_surface_form[column_name_embedder_input]=len(self.foreign_graph_in_surface_form)-1
        for i, table_name in enumerate(self.table_names):
            column_name_embedder_input = table_name.lower()
            self.foreign_graph_in.append(column_name_embedder_input)
            self.foreign_graph_in_id[column_name_embedder_input]=len(self.foreign_graph_in)-1
        self.key_link_edge1=[]
        self.key_link_edge2=[]
        
        foreign_keys=self.table_schema['foreign_keys']
        for key_pair in foreign_keys:
            k1=key_pair[0]
            k2=key_pair[1]
            c_name_sur1=self.column_names_original[k1][1].lower()
            c_name1=self.column_names[k1][1].lower()
            fa_id1=self.foreign_graph_in_id[self.table_names[self.column_names[k1][0]].lower()]


            c_name_sur2=self.column_names_original[k2][1].lower()
            c_name2=self.column_names[k2][1].lower()
            fa_id2=self.foreign_graph_in_id[self.table_names[self.column_names[k2][0]].lower()]


            if c_name_sur1 in self.foreign_graph_in_surface_form:
                id1=self.foreign_graph_in_id_surface_form[c_name_sur1]
            else:
                self.foreign_graph_in_surface_form.append(c_name_sur1)
                self.node_type.append('column')
                self.foreign_graph_in_id_surface_form[c_name_sur1]=len(self.foreign_graph_in_surface_form)-1

                self.foreign_graph_in.append(c_name1)
                self.foreign_graph_in_id[c_name1]=len(self.foreign_graph_in)-1
         
                id1=self.foreign_graph_in_id[c_name1]

        
            if c_name_sur2 in self.foreign_graph_in_surface_form:
                id2=self.foreign_graph_in_id_surface_form[c_name_sur2]
            else:
                self.foreign_graph_in_surface_form.append(c_name_sur2)
                self.node_type.append('column')
                self.foreign_graph_in_id_surface_form[c_name_sur2]=len(self.foreign_graph_in_surface_form)-1

                self.foreign_graph_in.append(c_name2)
                self.foreign_graph_in_id[c_name2]=len(self.foreign_graph_in)-1

                id2=self.foreign_graph_in_id[c_name2]
            
            if [fa_id1,id1] not in self.key_link_edge1:
                self.key_link_edge1.append([fa_id1,id1])
            
            if [fa_id2,id2] not in self.key_link_edge1:
                self.key_link_edge1.append([fa_id2,id2])

            if [id2,id1] not in self.key_link_edge2 and id2!=id1:
                
                self.key_link_edge2.append([id2,id1])
                self.key_link_edge2.append([id1,id2])

        assert len(self.foreign_graph_in)==len(self.foreign_graph_in_surface_form)
            
                

            


class Interaction:
    """ ATIS interaction class.

    Attributes:
        utterances (list of Utterance): The utterances in the interaction.
        snippets (list of Snippet): The snippets that appear through the interaction.
        anon_tok_to_ent:
        identifier (str): Unique identifier for the interaction in the dataset.
    """
    def __init__(self,
                 utterances,
                 schema,
                 snippets,
                 anon_tok_to_ent,
                 identifier,
                 params):
        self.utterances = utterances
        self.schema = schema
        self.snippets = snippets
        self.anon_tok_to_ent = anon_tok_to_ent
        self.identifier = identifier

        # Ensure that each utterance's input and output sequences, when remapped
        # without anonymization or snippets, are the same as the original
        # version.
        for i, utterance in enumerate(self.utterances):
            deanon_input = self.deanonymize(utterance.input_seq_to_use,
                                            ANON_INPUT_KEY)
            assert deanon_input == utterance.original_input_seq, "Anonymized sequence [" \
                + " ".join(utterance.input_seq_to_use) + "] is not the same as [" \
                + " ".join(utterance.original_input_seq) + "] when deanonymized (is [" \
                + " ".join(deanon_input) + "] instead)"
            desnippet_gold = self.expand_snippets(utterance.gold_query_to_use)
            deanon_gold = self.deanonymize(desnippet_gold, OUTPUT_KEY)
            assert deanon_gold == utterance.original_gold_query, \
                "Anonymized and/or snippet'd query " \
                + " ".join(utterance.gold_query_to_use) + " is not the same as " \
                + " ".join(utterance.original_gold_query)

    def __str__(self):
        string = "Utterances:\n"
        for utterance in self.utterances:
            string += str(utterance) + "\n"
        string += "Anonymization dictionary:\n"
        for ent_tok, deanon in self.anon_tok_to_ent.items():
            string += ent_tok + "\t" + str(deanon) + "\n"

        return string

    def __len__(self):
        return len(self.utterances)

    def deanonymize(self, sequence, key):
        """ Deanonymizes a predicted query or an input utterance.

        Inputs:
            sequence (list of str): The sequence to deanonymize.
            key (str): The key in the anonymization table, e.g. NL or SQL.
        """
        return anon.deanonymize(sequence, self.anon_tok_to_ent, key)

    def expand_snippets(self, sequence):
        """ Expands snippets for a sequence.

        Inputs:
            sequence (list of str): A SQL query.

        """
        return expand_snippets(sequence, self.snippets)

    def input_seqs(self):
        in_seqs = []
        for utterance in self.utterances:
            in_seqs.append(utterance.input_seq_to_use)
        return in_seqs

    def output_seqs(self):
        out_seqs = []
        for utterance in self.utterances:
            out_seqs.append(utterance.gold_query_to_use)
        return out_seqs

def load_function(parameters,
                  nl_to_sql_dict,
                  anonymizer,
                  database_schema=None):
    def fn(interaction_example):
        keep = False

        raw_utterances = interaction_example["interaction"]
   
        if "database_id" in interaction_example:
            database_id = interaction_example["database_id"]
            interaction_id = interaction_example["interaction_id"]
            identifier = database_id + '/' + str(interaction_id)
        else:
            identifier = interaction_example["id"]

        schema = None
        if database_schema:
            schema = Schema(database_schema[database_id], simple=True)

        snippet_bank = []

        utterance_examples = []

        anon_tok_to_ent = {}

        for utterance in raw_utterances:
            available_snippets = [
                snippet for snippet in snippet_bank if snippet.index <= 1]
        
            proc_utterance = Utterance(utterance,schema,
                                       available_snippets,
                                       nl_to_sql_dict,
                                       parameters,
                                       anon_tok_to_ent,
                                       anonymizer)
            keep_utterance = proc_utterance.keep

            if schema:
                assert keep_utterance

            if keep_utterance:
                keep = True
                utterance_examples.append(proc_utterance)

                # Update the snippet bank, and age each snippet in it.
                if parameters.use_snippets:
                    if 'atis' in parameters.data_directory:
                        snippets = sql_util.get_subtrees(
                            proc_utterance.anonymized_gold_query,
                            proc_utterance.available_snippets)
                    else:
                        snippets = sql_util.get_subtrees_simple(
                            proc_utterance.anonymized_gold_query,
                            proc_utterance.available_snippets)

                    for snippet in snippets:
                        snippet.assign_id(len(snippet_bank))
                        snippet_bank.append(snippet)

                for snippet in snippet_bank:
                    snippet.increase_age()

        interaction = Interaction(utterance_examples,
                                  schema,
                                  snippet_bank,
                                  anon_tok_to_ent,
                                  identifier,
                                  parameters)

        return interaction, keep

    return fn