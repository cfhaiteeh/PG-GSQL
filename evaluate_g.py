
import os
import sys

import numpy as np
import random

from parse_args import interpret_args

import data_util
from data_util import atis_data
from model.schema_interaction_model import SchemaInteractionATISModel
from logger import Logger
from model.model import ATISModel
from model_util1 import Metrics, evaluate_utterance_sample, evaluate_interaction_sample, \
    train_epoch_with_utterances, train_epoch_with_interactions, evaluate_using_predicted_queries

import torch

np.random.seed(0)
random.seed(0)

VALID_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
TRAIN_EVAL_METRICS = [Metrics.LOSS, Metrics.TOKEN_ACCURACY, Metrics.STRING_ACCURACY]
FINAL_EVAL_METRICS = [Metrics.STRING_ACCURACY, Metrics.TOKEN_ACCURACY]

def evaluate(model, data, params, savafile, split,step=1,first=None):
    """Evaluates a pretrained model on a dataset.

    Inputs:
        model (ATISModel): Model class.
        data (ATISData): All of the data.
        params (namespace): Parameters for the model.
        last_save_file (str): Location where the model save file is.
    """
    if savafile:
        model.load(savafile)
    else:
        if not params.save_file:
            raise ValueError(
                "Must provide a save file name if not training first.")
        model.load(params.save_file)

    filename = split

    if filename == 'dev':
        split = data.dev_data
    elif filename == 'train':
        split = data.train_data
    elif filename == 'test':
        split = data.test_data
    elif filename == 'valid':
        split = data.valid_data
    else:
        raise ValueError("Split not recognized: " + str(params.evaluate_split))

    if params.use_predicted_queries:
        filename += "_use_predicted_queries"
    else:
        filename += "_use_gold_queries"

    full_name = os.path.join(params.logdir, filename) + params.results_note


    examples = data.get_all_interactions(split)
    if step==1:
        _, _,first=evaluate_interaction_sample(
            examples,
            model,
            name=full_name,
            metrics=FINAL_EVAL_METRICS,
            total_num=atis_data.num_utterances(split),
            database_username=params.database_username,
            database_password=params.database_password,
            database_timeout=params.database_timeout,
            use_predicted_queries=params.use_predicted_queries,
            max_generation_length=params.eval_maximum_sql_length,
            write_results=True,
            use_gpu=True,
            compute_metrics=params.compute_metrics,step=step)
        return first
    elif step==2:
        evaluate_interaction_sample(
            examples,
            model,
            name=full_name,
            metrics=FINAL_EVAL_METRICS,
            total_num=atis_data.num_utterances(split),
            database_username=params.database_username,
            database_password=params.database_password,
            database_timeout=params.database_timeout,
            use_predicted_queries=params.use_predicted_queries,
            max_generation_length=params.eval_maximum_sql_length,
            write_results=True,
            use_gpu=True,
            compute_metrics=params.compute_metrics,step=step,first=first)
    else:
        evaluate_interaction_sample(
            examples,
            model,
            name=full_name,
            metrics=FINAL_EVAL_METRICS,
            total_num=atis_data.num_utterances(split),
            database_username=params.database_username,
            database_password=params.database_password,
            database_timeout=params.database_timeout,
            use_predicted_queries=params.use_predicted_queries,
            max_generation_length=params.eval_maximum_sql_length,
            write_results=True,
            use_gpu=True,
            compute_metrics=params.compute_metrics,step=step,first=first)



if __name__=='__main__':
    if os.path.exists('logs_sparc_pg_gsql/args.log'):
        os.remove('logs_sparc_pg_gsql/args.log')

    """Main function that trains and/or evaluates a model."""
    params = interpret_args()
 
    data = atis_data.ATISDataset(params)

    # Construct the model object.
    if params.interaction_level:
        model_type = SchemaInteractionATISModel
    else:
        print('not implemented')
        exit()
    if params.use_query_attention==1:
        params.use_query_attention=1
        model = model_type(
            params,
            data.input_vocabulary,
            data.output_vocabulary,
            data.output_vocabulary_schema,
            data.anonymizer if params.anonymize and params.anonymization_scoring else None)
        model = model.cuda()
        evaluate(model, data, params, 'sparc_pg_gsql_paper_save/save_opt', split='valid',step=2,first=None)
        



    else:
        model = model_type(
            params,
            data.input_vocabulary,
            data.output_vocabulary,
            data.output_vocabulary_schema,
            data.anonymizer if params.anonymize and params.anonymization_scoring else None)
        model = model.cuda()
        evaluate(model, data, params, 'spider_bert_530.4/save_12', split='valid',step=2,first=None)

