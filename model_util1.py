"""Basic model training and evaluation functions."""

from enum import Enum

import random
import sys
import copy
import progressbar
import json

from model import torch_utils
from data_util import sql_util
import torch
fgf = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except','end','having','desc')

xrc=['avg','min','max','sum','count']
def convert2fine(SQL_toks,input_schema):
    all_cols = [x[1].lower() for x in input_schema.column_names_original]

    select_flag=0
    stack=[]
    ans_sql=[]
    SQL_toks.append('end')
    for tk in SQL_toks:
        if tk=='select':
            select_flag=1
            ans_sql.append('select')
        if tk!='select' and tk in fgf:
            if len(ans_sql)>0 and ans_sql[-1]==',':
                ans_sql = ans_sql[:-1]
            select_flag=0

        if tk=='whole':
            ans_sql.append('count')
            ans_sql.append('(')
            ans_sql.append('*')
            ans_sql.append(')')
            if select_flag!=0:
                ans_sql.append(',')

        if select_flag==1:
            if tk in xrc:
                stack.append(tk)
            if tk in all_cols:
                if len(stack)==0:
                    ans_sql.append(tk)
                    ans_sql.append(',')
                else:
                    for xx in stack:
                        ans_sql.append(xx)
                        ans_sql.append('(')
                        ans_sql.append(tk)
                        ans_sql.append(')')
                        ans_sql.append(',')
        elif tk !='whole':
            ans_sql.append(tk)
    return ans_sql[:-1]
CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except','end')
SQL_OPS = ('intersect', 'union', 'except')
def list_swap(sql_toks, lr1, lr2):
  tk1 = sql_toks[lr1[0]:lr1[1]]
  tk2 = sql_toks[lr2[0]:lr2[1]]
  sql_toks[lr1[0]:lr1[0] + len(tk2)] = tk2
  sql_toks[lr1[0] + len(tk2):lr1[0] + len(tk2) + len(tk1)] = tk1
  return sql_toks
def revise(sqltok):
  sqltok.append('end')
  s = 0
  for x in sqltok:
    if x in SQL_OPS:
      s += 1

  t = (sqltok.count('from'))
  t1 = (sqltok.count('select'))

  s_list = []
  l = -1
  for idx, x in enumerate(sqltok):

    if l != -1 and x in CLAUSE_KEYWORDS:
      s_list.append([l, idx])
      l = -1
    if x == 'from':
      l = idx

  f_list = []
  l = -1
  zkh=0
  clause=-1
  for idx, x in enumerate(sqltok):
    if zkh==0:
      clause=0
    if x=='(':
      zkh+=1
    if x==')':
      zkh-=1
    if zkh>0 and x=='from':
      clause=1


    if l != -1 and (x in CLAUSE_KEYWORDS or (zkh==0 and clause==1)):
      f_list.append([l, idx])
      l = -1
    if x == 'select':
      l = idx
  for lr1, lr2 in zip(s_list, f_list):
    sqltok = list_swap(sqltok, lr1, lr2)
  return sqltok[0:-1]
def t1_t2_generate(psql,schema):
  tabs = [x.lower() for x in schema.table_names_original]

  icols=schema.column_names_original
  cols = [x[1].lower() for x in schema.column_names_original]
  cols_tb = [x[0] for x in schema.column_names_original]
  psql.append('end')
  CLAUSE_KEYWORDS1 = ('select',  'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'end')

  t_t=['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10']
  n_t=0
  ans=[]
  flag=0
  tbind=[]

  foreign_keys=schema.foreign_keys


  for tk in psql:
    if flag == 1 and tk in CLAUSE_KEYWORDS1:
      if n_t == 1:
        ans = ans[:-2]
      flag = 0
      n_t = 0
      tbind = []
    if flag == 1 and tk in tabs:
      if n_t > 0:
        ans.append('join')
    ans.append(tk)
    if tk=='from':
      flag=1

    if flag==1 and tk in tabs:
      tbind.append(tabs.index(tk))

      ans.append('as')
      ans.append(t_t[n_t])
      if n_t>0:
        ans.append('on')
        id1=tbind[-2]
        id2=tbind[-1]
        ok=0

     
        link_flag=[]
        for k1,k2 in foreign_keys:
            if cols_tb[k1] in tbind and cols_tb[k2] in tbind:
                
                link_flag.append(cols_tb[k1] )
                link_flag.append(cols_tb[k2] )

        for x,y in icols:
          if x==id1:

            for x1,y1 in icols:
              if x1==id2:
                  if y.lower()==y1.lower():
                      link_flag.append(x)
                      link_flag.append(y)
        fo_flag=1
        if len(tbind)>1:   
            for _t_id in tbind:

                if _t_id not in link_flag:
                    fo_flag=0
                    break
        if fo_flag==0:
            psql.append('foreign')
            psql.append('error')
            return psql

        for x,y in icols:
          if x==id1:

            for x1,y1 in icols:
              if x1==id2:
                ok=1



                ans.append(str(t_t[len(tbind)-2])+'.'+y.lower())
                ans.append('=')
                ans.append(str(t_t[len(tbind)-1])+'.'+y1.lower())

                break
          if ok==1:
            break

      n_t+=1
  over=[]
  flag=0
  ttmap=dict()
  n_t=0
  for idx,tk in enumerate(ans):
    if flag == 1 and tk in CLAUSE_KEYWORDS1:
      flag = 0

    if tk == 'from':
      flag = 1
      ttmap = dict()
      n_t=0

    if flag == 1 and tk in tabs:
      ttmap[tk] = t_t[n_t]
      n_t+=1
    pre=''
    if flag == 0 and tk !='*' and tk in cols and n_t>1:
      for x, y in icols:
        if y.lower() == tk and tabs[x] in ttmap:
          pre=ttmap[tabs[x]]+'.'
          break
    over.append(pre+tk)




  return over[:-1]

def del_tab(sql_tks,input_schema):

    if '*' in sql_tks:
        return sql_tks
    tabs = [x.lower() for x in input_schema.table_names_original]

    CLAUSE_KEYWORDS1 = ('select',  'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'end')
    flag=0
    user_tb=[]

    for tk in sql_tks:
            if tk=='from':
                flag=1
            if flag==1 and tk in CLAUSE_KEYWORDS1:
                flag=0
            if flag==0:
                for x,y in input_schema.column_names_original:
                    if x==-1:
                        continue
                    y=y.lower()
                    if y==tk:
                        user_tb.append(tabs[x])
    new_tks=[]
    flag=0
    for x in sql_tks:
        if x=='from':
            flag=1
        if flag==1 and x in CLAUSE_KEYWORDS1:
            flag=0
        
        if flag==1 and  x in tabs and  x not in user_tb:
            continue

        new_tks.append(x)

    return new_tks


def key_gen(sql_toks,input_schema):
    if 'in' not in sql_toks:
        return sql_toks
    
    sp_id=sql_toks.index('in')

    cols = [x[1].lower() for x in input_schema.column_names_original]
    colsID = [x[0] for x in input_schema.column_names_original]

    tabs = [x.lower() for x in input_schema.table_names_original]
    foreign_keys=input_schema.foreign_keys
    tb1=[]
    co1=''
    id1=0
    idx1=0
    for tk in sql_toks[0:sp_id]:
        
        if tk in tabs:
            tb1.append(tk)
        if tk in cols:
            co1=tk
            id1=idx1
        idx1+=1
        
    tb2=[]
    co2=''
    id2=0
    idx2=sp_id
    for tk in sql_toks[sp_id:]:
        if tk in tabs:
            tb2.append(tk)
        if tk in cols and co2=='':
            co2=tk
            id2=idx2
        idx2+=1
    if co1==co2:
        return sql_toks
    
    for x,y in foreign_keys:
        c1=cols[x]
        c2=cols[y]
        if (c1==co1 or c2==co1) and (c1==co2 or c2==co2):
            return sql_toks
    
    for x,y in foreign_keys:
        c1=cols[x]
        c2=cols[y]
        if c1==co1:
            if tabs[colsID[x]] in tb1 and tabs[colsID[y]] in tb2:
                sql_toks[id2]=c2
                return sql_toks
        
        if c2==co1:
            if tabs[colsID[y]] in tb1 and tabs[colsID[x]] in tb2:
                sql_toks[id2]=c1
                return sql_toks
        if c1==co2:
            if tabs[colsID[x]] in tb2 and tabs[colsID[y]] in tb1:
                sql_toks[id1]=c2
                return sql_toks
        
        if c2==co2:
            if tabs[colsID[y]] in tb2 and tabs[colsID[x]] in tb1:
                sql_toks[id1]=c1
                return sql_toks

    return sql_toks
        



def write_prediction(fileptr,
                     identifier,
                     input_seq,
                     probability,
                     prediction,
                     flat_prediction,
                     gold_query,
                     flat_gold_queries,
                     gold_tables,
                     index_in_interaction,
                     database_username,
                     database_password,
                     database_timeout,
                     compute_metrics=True,input_schema=None,step=1):
    pred_obj = {}
    pred_obj["identifier"] = identifier
    if len(identifier.split('/')) == 2:
        database_id, interaction_id = identifier.split('/')
    else:
        database_id = 'atis'
        interaction_id = identifier
    pred_obj["database_id"] = database_id
    pred_obj["interaction_id"] = interaction_id

    pred_obj["input_seq"] = input_seq
    pred_obj["probability"] = probability
    pred_obj["prediction"] = prediction

    new_flat_prediction=[]
    for x in flat_prediction:
        t=x.replace('table ','')
        t=t.replace('column ','')
        new_flat_prediction.append(t)

    if step==-1:
        pred_obj["flat_prediction"] = new_flat_prediction
    else:
        try:
          
            pred_obj["flat_prediction"] =new_flat_prediction
            pred_obj["flat_prediction"] = t1_t2_generate(pred_obj["flat_prediction"] ,input_schema)
            pred_obj["flat_prediction"] =revise(pred_obj["flat_prediction"] )
        except:
            pred_obj["flat_prediction"] = new_flat_prediction
            
    pred_obj["gold_query"] = gold_query
    pred_obj["flat_gold_queries"] = flat_gold_queries
    pred_obj["index_in_interaction"] = index_in_interaction
    pred_obj["gold_tables"] = str(gold_tables)
    if compute_metrics:
        correct_string = " ".join(flat_prediction) in [
            " ".join(q) for q in flat_gold_queries]
        pred_obj["correct_string"] = correct_string

        if not correct_string:
            syntactic, semantic, pred_table = sql_util.execution_results(
                " ".join(flat_prediction), database_username, database_password, database_timeout)
            pred_table = sorted(pred_table)
            best_prec = 0.
            best_rec = 0.
            best_f1 = 0.

            for gold_table in gold_tables:
                num_overlap = float(len(set(pred_table) & set(gold_table)))

                if len(set(gold_table)) > 0:
                    prec = num_overlap / len(set(gold_table))
                else:
                    prec = 1.

                if len(set(pred_table)) > 0:
                    rec = num_overlap / len(set(pred_table))
                else:
                    rec = 1.

                if prec > 0. and rec > 0.:
                    f1 = (2 * (prec * rec)) / (prec + rec)
                else:
                    f1 = 1.

                best_prec = max(best_prec, prec)
                best_rec = max(best_rec, rec)
                best_f1 = max(best_f1, f1)

        else:
            syntactic = True
            semantic = True
            pred_table = []
            best_prec = 1.
            best_rec = 1.
            best_f1 = 1.

        assert best_prec <= 1.
        assert best_rec <= 1.
        assert best_f1 <= 1.
        pred_obj["syntactic"] = syntactic
        pred_obj["semantic"] = semantic
        correct_table = (pred_table in gold_tables) or correct_string
        pred_obj["correct_table"] = correct_table
        pred_obj["strict_correct_table"] = correct_table and syntactic
        pred_obj["pred_table"] = str(pred_table)
        pred_obj["table_prec"] = best_prec
        pred_obj["table_rec"] = best_rec
        pred_obj["table_f1"] = best_f1

    fileptr.write(json.dumps(pred_obj) + "\n")
    return pred_obj["flat_prediction"]

class Metrics(Enum):
    """Definitions of simple metrics to compute."""
    LOSS = 1
    TOKEN_ACCURACY = 2
    STRING_ACCURACY = 3
    CORRECT_TABLES = 4
    STRICT_CORRECT_TABLES = 5
    SEMANTIC_QUERIES = 6
    SYNTACTIC_QUERIES = 7


def get_progressbar(name, size):
    """Gets a progress bar object given a name and the total size.

    Inputs:
        name (str): The name to display on the side.
        size (int): The maximum size of the progress bar.

    """
    return progressbar.ProgressBar(maxval=size,
                                   widgets=[name,
                                            progressbar.Bar('=', '[', ']'),
                                            ' ',
                                            progressbar.Percentage(),
                                            ' ',
                                            progressbar.ETA()])


def train_epoch_with_utterances(batches,
                                model,
                                randomize=True):
    """Trains model for a single epoch given batches of utterance data.

    Inputs:
        batches (UtteranceBatch): The batches to give to training.
        model (ATISModel): The model obect.
        learning_rate (float): The learning rate to use during training.
        dropout_amount (float): Amount of dropout to set in the model.
        randomize (bool): Whether or not to randomize the order that the batches are seen.
    """
    if randomize:
        random.shuffle(batches)
    progbar = get_progressbar("train     ", len(batches))
    progbar.start()
    loss_sum = 0.

    for i, batch in enumerate(batches):
        batch_loss = model.train_step(batch)
        loss_sum += batch_loss

        progbar.update(i)

    progbar.finish()

    total_loss = loss_sum / len(batches)

    return total_loss


def train_epoch_with_interactions(interaction_batches,
                                  params,
                                  model,
                                  randomize=True,epoch=-1000):
    """Trains model for single epoch given batches of interactions.

    Inputs:
        interaction_batches (list of InteractionBatch): The batches to train on.
        params (namespace): Parameters to run with.
        model (ATISModel): Model to train.
        randomize (bool): Whether or not to randomize the order that batches are seen.
    """
    if randomize:
        random.shuffle(interaction_batches)
    progbar = get_progressbar("train     ", len(interaction_batches))
    progbar.start()
    loss_sum = 0.

    for i, interaction_batch in enumerate(interaction_batches):
        assert len(interaction_batch) == 1
        interaction = interaction_batch.items[0]

        if interaction.identifier == "raw/atis2/12-1.1/ATIS2/TEXT/TEST/NOV92/770/5":
            continue
        if "baseball_1" in interaction.identifier:
            continue
     
        batch_loss = model.train_step(interaction, params.train_maximum_sql_length,epoch=epoch)

        loss_sum += batch_loss
        torch.cuda.empty_cache()

        progbar.update(i)

    progbar.finish()

    total_loss = loss_sum / len(interaction_batches)

    return total_loss


def update_sums(metrics,
                metrics_sums,
                predicted_sequence,
                flat_sequence,
                gold_query,
                original_gold_query,
                gold_forcing=False,
                loss=None,
                token_accuracy=0.,
                database_username="",
                database_password="",
                database_timeout=0,
                gold_table=None):
    """" Updates summing for metrics in an aggregator.

    TODO: don't use sums, just keep the raw value.
    """
    if Metrics.LOSS in metrics :
        metrics_sums[Metrics.LOSS] += loss.item()
    if Metrics.TOKEN_ACCURACY in metrics:
        if gold_forcing:
            metrics_sums[Metrics.TOKEN_ACCURACY] += token_accuracy
        else:
            num_tokens_correct = 0.
            for j, token in enumerate(gold_query):
                if len(
                        predicted_sequence) > j and predicted_sequence[j] == token:
                    num_tokens_correct += 1
            metrics_sums[Metrics.TOKEN_ACCURACY] += num_tokens_correct / \
                len(gold_query)
    if Metrics.STRING_ACCURACY in metrics:
        x=copy.deepcopy(flat_sequence)
        y=copy.deepcopy(original_gold_query)
        x.sort()
        y.sort()
        metrics_sums[Metrics.STRING_ACCURACY] += int(
            x==y)

    if Metrics.CORRECT_TABLES in metrics:
        assert database_username, "You did not provide a database username"
        assert database_password, "You did not provide a database password"
        assert database_timeout > 0, "Database timeout is 0 seconds"

        if flat_sequence != original_gold_query:
            syntactic, semantic, table = sql_util.execution_results(
                " ".join(flat_sequence), database_username, database_password, database_timeout)
        else:
            syntactic = True
            semantic = True
            table = gold_table

        metrics_sums[Metrics.CORRECT_TABLES] += int(table == gold_table)
        if Metrics.SYNTACTIC_QUERIES in metrics:
            metrics_sums[Metrics.SYNTACTIC_QUERIES] += int(syntactic)
        if Metrics.SEMANTIC_QUERIES in metrics:
            metrics_sums[Metrics.SEMANTIC_QUERIES] += int(semantic)
        if Metrics.STRICT_CORRECT_TABLES in metrics:
            metrics_sums[Metrics.STRICT_CORRECT_TABLES] += int(
                table == gold_table and syntactic)


def construct_averages(metrics_sums, total_num):
    """ Computes the averages for metrics.

    Inputs:
        metrics_sums (dict Metric -> float): Sums for a metric.
        total_num (int): Number to divide by (average).
    """
    metrics_averages = {}
    for metric, value in metrics_sums.items():
        metrics_averages[metric] = value / total_num
        if metric != "loss":
            metrics_averages[metric] *= 100.

    return metrics_averages


def evaluate_utterance_sample(sample,
                              model,
                              max_generation_length,
                              name="",
                              gold_forcing=False,
                              metrics=None,
                              total_num=-1,
                              database_username="",
                              database_password="",
                              database_timeout=0,
                              write_results=False):
    """Evaluates a sample of utterance examples.

    Inputs:
        sample (list of Utterance): Examples to evaluate.
        model (ATISModel): Model to predict with.
        max_generation_length (int): Maximum length to generate.
        name (str): Name to log with.
        gold_forcing (bool): Whether to force the gold tokens during decoding.
        metrics (list of Metric): Metrics to evaluate with.
        total_num (int): Number to divide by when reporting results.
        database_username (str): Username to use for executing queries.
        database_password (str): Password to use when executing queries.
        database_timeout (float): Timeout on queries when executing.
        write_results (bool): Whether to write the results to a file.
    """
    assert metrics

    if total_num < 0:
        total_num = len(sample)

    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.

    predictions_file = open(name + "_predictions.json", "w")
    print("Predicting with filename " + str(name) + "_predictions.json")
    progbar = get_progressbar(name, len(sample))
    progbar.start()

    predictions = []
    for i, item in enumerate(sample):
        _, loss, predicted_seq = model.eval_step(
            item, max_generation_length, feed_gold_query=gold_forcing)
        loss = loss / len(item.gold_query())
        predictions.append(predicted_seq)

        flat_sequence = item.flatten_sequence(predicted_seq)
        token_accuracy = torch_utils.per_token_accuracy(
            item.gold_query(), predicted_seq)

        if write_results:
            write_prediction(
                predictions_file,
                identifier=item.interaction.identifier,
                input_seq=item.input_sequence(),
                probability=0,
                prediction=predicted_seq,
                flat_prediction=flat_sequence,
                gold_query=item.gold_query(),
                flat_gold_queries=item.original_gold_queries(),
                gold_tables=item.gold_tables(),
                index_in_interaction=item.utterance_index,
                database_username=database_username,
                database_password=database_password,
                database_timeout=database_timeout)

        update_sums(metrics,
                    metrics_sums,
                    predicted_seq,
                    flat_sequence,
                    item.gold_query(),
                    item.original_gold_queries()[0],
                    gold_forcing,
                    loss,
                    token_accuracy,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    gold_table=item.gold_tables()[0])

        progbar.update(i)

    progbar.finish()
    predictions_file.close()

    return construct_averages(metrics_sums, total_num), None


def evaluate_interaction_sample(sample,
                                model,
                                max_generation_length,
                                name="",
                                gold_forcing=False,
                                metrics=None,
                                total_num=-1,
                                database_username="",
                                database_password="",
                                database_timeout=0,
                                use_predicted_queries=False,
                                write_results=False,
                                use_gpu=False,
                                compute_metrics=False,all_pre=None,step=1,first=None):
    """ Evaluates a sample of interactions. """
    predictions_file = open(name + "_predictions.json", "w")
    print("Predicting with file " + str(name + "_predictions.json"))
    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.
    progbar = get_progressbar(name, len(sample))
    progbar.start()

    num_utterances = 0
    predictions = []

    use_gpu = not ("--no_gpus" in sys.argv or "--no_gpus=1" in sys.argv)

    model.eval()
    if step==1 or step==3:
        all_pre=[[0]]
        first=[]
    else:
        all_pre=first
    from_json=[]
    f=open('countries')
    allcountries=[]
    for line in f:
        allcountries.append(line.strip().lower())
    f.close()
    for i, interaction in enumerate(sample):
        with torch.no_grad():
            if use_predicted_queries:
                example_preds,all_pre,input_schema = model.predict_with_predicted_queries(
                    interaction,
                    max_generation_length,from_json,all_pre=all_pre,allcountries=allcountries)
            else:
                example_preds,all_pre,input_schema = model.predict_with_gold_queries(
                    interaction,
                    max_generation_length,
                    feed_gold_query=gold_forcing,all_pre=all_pre,allcountries=allcountries)
            torch.cuda.empty_cache()

        predictions.extend(example_preds)

        assert len(example_preds) == len(
            interaction.interaction.utterances) or not example_preds
        for j, pred in enumerate(example_preds):
            num_utterances += 1

            sequence, loss, token_accuracy, _, decoder_results = pred


            if use_predicted_queries:
                item = interaction.processed_utterances[j]
                original_utt = interaction.interaction.utterances[item.index]

                gold_query = original_utt.gold_query_to_use
                original_gold_query = original_utt.original_gold_query

                gold_table = original_utt.gold_sql_results
                gold_queries = [q[0] for q in original_utt.all_gold_queries]
                gold_tables = [q[1] for q in original_utt.all_gold_queries]
                index = item.index
            else:
                item = interaction.gold_utterances()[j]

                gold_query = item.gold_query()
                original_gold_query = item.original_gold_query()

                gold_table = item.gold_table()
                gold_queries = item.original_gold_queries()
                gold_tables = item.gold_tables()
                index = item.utterance_index
            if loss:
                loss = loss / len(gold_query)

            flat_sequence = item.flatten_sequence(sequence)

            if write_results:
                pre_q=write_prediction(
                    predictions_file,
                    identifier=interaction.identifier,
                    input_seq=item.input_sequence(),
                    probability=decoder_results.probability,
                    prediction=sequence,
                    flat_prediction=flat_sequence,
                    gold_query=gold_query,
                    flat_gold_queries=gold_queries,
                    gold_tables=gold_tables,
                    index_in_interaction=index,
                    database_username=database_username,
                    database_password=database_password,
                    database_timeout=database_timeout,
                    compute_metrics=compute_metrics,input_schema=input_schema,step=step)
                if step==1:
                    first.append(pre_q)
            update_sums(metrics,
                        metrics_sums,
                        sequence,
                        flat_sequence,
                        gold_query,
                        original_gold_query,
                        gold_forcing,
                        loss,
                        token_accuracy,
                        database_username=database_username,
                        database_password=database_password,
                        database_timeout=database_timeout,
                        gold_table=gold_table)

        progbar.update(i)

    progbar.finish()

    if total_num < 0:
        total_num = num_utterances

    predictions_file.close()
    return construct_averages(metrics_sums, total_num), predictions,first


def evaluate_using_predicted_queries(sample,
                                     model,
                                     name="",
                                     gold_forcing=False,
                                     metrics=None,
                                     total_num=-1,
                                     database_username="",
                                     database_password="",
                                     database_timeout=0,
                                     snippet_keep_age=1):
    predictions_file = open(name + "_predictions.json", "w")
    print("Predicting with file " + str(name + "_predictions.json"))
    assert not gold_forcing
    metrics_sums = {}
    for metric in metrics:
        metrics_sums[metric] = 0.
    progbar = get_progressbar(name, len(sample))
    progbar.start()

    num_utterances = 0
    predictions = []
    for i, item in enumerate(sample):
        int_predictions = []
        item.start_interaction()
        while not item.done():
            utterance = item.next_utterance(snippet_keep_age)

            predicted_sequence, loss, _, probability = model.eval_step(
                utterance)
            int_predictions.append((utterance, predicted_sequence))

            flat_sequence = utterance.flatten_sequence(predicted_sequence)

            if sql_util.executable(
                    flat_sequence,
                    username=database_username,
                    password=database_password,
                    timeout=database_timeout) and probability >= 0.24:
                utterance.set_pred_query(
                    item.remove_snippets(predicted_sequence))
                item.add_utterance(utterance,
                                   item.remove_snippets(predicted_sequence),
                                   previous_snippets=utterance.snippets())
            else:
                seq = []
                utterance.set_pred_query(seq)
                item.add_utterance(
                    utterance, seq, previous_snippets=utterance.snippets())

            original_utt = item.interaction.utterances[utterance.index]
            write_prediction(
                predictions_file,
                identifier=item.interaction.identifier,
                input_seq=utterance.input_sequence(),
                probability=probability,
                prediction=predicted_sequence,
                flat_prediction=flat_sequence,
                gold_query=original_utt.gold_query_to_use,
                flat_gold_queries=[
                    q[0] for q in original_utt.all_gold_queries],
                gold_tables=[
                    q[1] for q in original_utt.all_gold_queries],
                index_in_interaction=utterance.index,
                database_username=database_username,
                database_password=database_password,
                database_timeout=database_timeout)

            update_sums(metrics,
                        metrics_sums,
                        predicted_sequence,
                        flat_sequence,
                        original_utt.gold_query_to_use,
                        original_utt.original_gold_query,
                        gold_forcing,
                        loss,
                        token_accuracy=0,
                        database_username=database_username,
                        database_password=database_password,
                        database_timeout=database_timeout,
                        gold_table=original_utt.gold_sql_results)

        predictions.append(int_predictions)
        progbar.update(i)

    progbar.finish()

    if total_num < 0:
        total_num = num_utterances
    predictions_file.close()

    return construct_averages(metrics_sums, total_num), predictions
