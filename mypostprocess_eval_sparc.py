import os
import json
import argparse
import subprocess
idx=['table_names_original','column_names_original']
used=['_UNK', '_EOS', '.', 't1', 'select', 'from', 't2', '=', 'as', 'value', ')', '(', 'where', 'join', 'on', 'by', ',', 'count', 't3', 'group', 'order', 'limit', 'desc', '>', 'distinct', 'and', 'avg', 'having', '<', 'in', 'sum', 'max', 'asc', 'not', 't4', 'or', 'like', 'min', 'intersect', 'except', '!', 'union', 'between', 't5', '-', '+']
import copy
import nltk

def t1_t2_generate(psql,schema):
  tabs = [x.lower() for x in schema['table_names_original']]


  icols=schema['column_names_original']
  cols = [x[1].lower() for x in schema['column_names_original']]

  psql.append('end')
  CLAUSE_KEYWORDS1 = ('select',  'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'end')

  t_t=['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10']
  n_t=0
  ans=[]
  flag=0
  tbind=[]
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





def postprocess(predictions, database_schema):
  
  total = 0
  postprocess_sqls = {}
 



  s0=0



  for pred in predictions:


    db_id = pred['database_id']
 


 
    if db_id not in postprocess_sqls:
      postprocess_sqls[db_id] = []

    interaction_id = pred['interaction_id']
    turn_id = pred['index_in_interaction']
    total += 1
    wt=[w.lower() for w in pred['input_seq']]
    wt=' '.join(wt)

    s0+=1



    pred_sql_str = ' '.join(pred['flat_prediction']).lower()


 
    postprocess_sql = pred_sql_str

    postprocess_sqls[db_id].append((postprocess_sql, interaction_id, turn_id,wt))

  return postprocess_sqls

def read_prediction(pred_file):
  print('Read prediction from', pred_file)
  predictions = []
  with open(pred_file) as f:
    for line in f:
      pred = json.loads(line)

      predictions.append(pred)
  print('Number of predictions', len(predictions))
  return predictions

def read_schema(table_schema_path):
  with open(table_schema_path) as f:
    database_schema = json.load(f)

  database_schema_dict = {}
  for table_schema in database_schema:
    db_id = table_schema['db_id']
    database_schema_dict[db_id] = table_schema

  return database_schema_dict

def write_and_evaluate(postprocess_sqls, db_path, table_schema_path, gold_path, dataset):
  db_list = []
  with open(gold_path) as f:
    for line in f:
      line_split = line.strip().split('\t')
      if len(line_split) != 2:
        continue
      db = line.strip().split('\t')[1]
      if db not in db_list:
        db_list.append(db)

  if dataset == 'sparc':
    cnt = 0
    output_file = 'output_temp.txt'
    with open(output_file, "w") as f:
      for db in db_list:
        for postprocess_sql, interaction_id, turn_id,wt in postprocess_sqls[db]:
          if turn_id == 0 and cnt > 0:
            f.write('\n')
          f.write('{}\n'.format(postprocess_sql))
          cnt += 1
    output_file1 = 'output_temp1.txt'
    cnt = 0
    with open(output_file1, "w") as f:
      for db in db_list:
        for postprocess_sql, interaction_id, turn_id, wt in postprocess_sqls[db]:
          if turn_id == 0 and cnt > 0:
            f.write('\n')
          f.write(wt+'\n')
          cnt += 1

    command = 'python2 ansev.py --db {} --table {} --etype match --gold {} --pred {} --wt {}'.format(db_path,
                                                                                                      table_schema_path,
                                                                                                      gold_path,
                                                                                                      os.path.abspath(output_file),os.path.abspath(output_file1))
    command += '; rm output_temp.txt'
  else:
    cnt = 0
    output_file = 'output_temp.txt'
    with open(output_file, "w") as f:
      for db in db_list:
        for postprocess_sql, interaction_id, turn_id,wt in postprocess_sqls[db]:
  
          if turn_id == 0 and cnt > 0:
            f.write('\n')
          f.write('{}\n'.format(postprocess_sql))
          cnt += 1
    output_file1 = 'output_temp1.txt'
    cnt = 0
    with open(output_file1, "w") as f:
      for db in db_list:
        for postprocess_sql, interaction_id, turn_id, wt in postprocess_sqls[db]:
      
          if turn_id == 0 and cnt > 0:

            f.write('\n')
          f.write(wt+'\n')
          cnt += 1

    command = 'python2 ansev.py --db {} --table {} --etype match --gold {} --pred {} --wt {}'.format(db_path,
                                                                                                      table_schema_path,
                                                                                                      gold_path,
                                                                                                      os.path.abspath(output_file),os.path.abspath(output_file1))
    command += '; rm output_temp.txt'
  return command



CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except','end')
SQL_OPS = ('intersect', 'union', 'except')

def merge(sqltok,schema):
  tabs=[x.lower() for x in schema['table_names_original']]


  cols = schema['column_names_original']


  for idx,x in enumerate(sqltok):
    if x=='from' and sqltok[idx+1] in tabs:
      usedtb=sqltok[idx+1]
    else:
      usedtb=''




def list_swap(sql_toks, lr1, lr2):
  tk1 = sql_toks[lr1[0]:lr1[1]]
  tk2 = sql_toks[lr2[0]:lr2[1]]
  sql_toks[lr1[0]:lr1[0] + len(tk2)] = tk2
  sql_toks[lr1[0] + len(tk2):lr1[0] + len(tk2) + len(tk1)] = tk1
  return sql_toks

def error_remove(sqltok,schema):

  tabs=schema['table_names_original']
  cols=schema['column_names_original']

  tb_i=dict()

  try:
    for idx,x in enumerate(sqltok):
      if x=='as':
        tb_i[sqltok[idx+1]]=sqltok[idx-1]
      if x=='.':
        tb_name=tb_i[sqltok[idx-1]]
        co_name=sqltok[idx+1]
   
        it=-1
        for ti,t_ in enumerate(tabs):
          if tb_name==t_.lower():
            it=ti
            break
        c_flag=0
        for i_c in cols:
          if i_c[1].lower()==co_name and i_c[0]==it:
            c_flag=1
            break
        if c_flag==0:

            if sqltok[idx-1]=='t1':
              sqltok[idx - 1]='t2'
            else:
              sqltok[idx - 1]='t1'
  except:
      print('ee')



  return sqltok



def sql_error_remove(sqltok):

  l=-1
  t_used=[]

  for idx,x in enumerate(sqltok):
    if l==-1:
      if x=='t1' or x=='t2' or x=='t3':
        if x not in t_used:
          t_used.append(x)


    if l != -1 and x in CLAUSE_KEYWORDS:
      l = -1
    if x == 'from':
      l = idx

  if len(t_used)!=1 :
    return sqltok

  fromx=''
  for idx,x in enumerate(sqltok):
    if x=='as':
      if sqltok[idx+1]==t_used[0]:
        fromx=sqltok[idx-1]
        break

  anstok=[]

  for idx,x in enumerate(sqltok):
    if l != -1 and x in CLAUSE_KEYWORDS:
      l = -1
    if x ==t_used[0] or x=='.' or l!=-1:
      continue

    if x == 'from':
      l = idx
      anstok.append('from')
      anstok.append(fromx)
    else:
      anstok.append(x)

  return anstok





def revise(sqltok):

  sqltok.append('end')
  s = 0
  for x in sqltok:
    if x in SQL_OPS:
      s += 1

 

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






import inflect
pex = inflect.engine()
def getcmp(sql1,schema):
  tabs = [x.lower() for x in schema['table_names_original']]

  atabs = [x.lower() for x in schema['table_names']]

  cols = [x[1].lower() for x in schema['column_names_original']]
  acols = [x[1].lower() for x in schema['column_names']]
  cmp=[]
  for x in sql1:
    if x in tabs and atabs[tabs.index(x)] not in cmp:
      cmp.append(atabs[tabs.index(x)])
    elif x in cols and acols[cols.index(x)] not in cmp:
      cmp.append(acols[cols.index(x)])
  return cmp

def calsum(cmp,seq):
  sum=0
  for x in cmp:
    x=x.split()
    ok=0
    for idx,y in enumerate(seq):
      if idx+len(x)<=len(seq):
        z=seq[idx:idx+len(x)]
        flag = 1
        for c1,c2 in zip(x,z):
          c1=c1.lower()
          c2=c2.lower()
          if pex.singular_noun(c1) != False:
            c1 = pex.singular_noun(c1)
          if pex.singular_noun(c2) != False:
            c2 = pex.singular_noun(c2)
          if c1!=c2:
            flag=0
            break
        if flag==1:
          ok=1
          break
    sum+=ok
  return sum



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='cosql')
  parser.add_argument('--split', type=str, default='dev')
  parser.add_argument('--pred_file', type=str, default='')
  parser.add_argument('--mypred_file', type=str, default='')

  args = parser.parse_args()

  if args.dataset == 'sparc':
    db_path = 'data/database/'
    table_schema_path = 'data/sparc/tables.json'

    if args.split == 'dev':
      gold_path = 'data/sparc/dev_gold.txt'
    pred_file = 'logs_sparc_cdseq2seq/valid_use_predicted_queries_predictions.json'
    
  elif args.dataset == 'spider':
 
    db_path = 'data/database/'

    table_schema_path = 'data/spider/tables.json'

    if args.split == 'dev':
      gold_path = 'data/spider/dev_gold.sql'
    pred_file = 'logs_spider_cdseq2seq/valid_use_predicted_queries_predictions.json'
  elif args.dataset == 'cosql':
  
    db_path = 'data/database/'

    table_schema_path = 'data/cosql_data/tables.json'

    if args.split == 'dev':
      gold_path = 'data/cosql/dev_gold.txt'
    pred_file = 'logs_cosql_cdseq2seq/valid_use_predicted_queries_predictions.json'


  database_schema = read_schema(table_schema_path)
  predictions = read_prediction(pred_file)
 
  postprocess_sqls = postprocess(predictions, database_schema)

  command = write_and_evaluate(postprocess_sqls, db_path, table_schema_path, gold_path, args.dataset)


  eval_output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
  with open(pred_file+'.eval', 'w') as f:
    f.write(eval_output.decode("utf-8"))
  print('Eval result in', pred_file+'.eval')




