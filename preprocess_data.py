
import os
import pickle
import json
import shutil

import copy

from nltk import word_tokenize

import  new_i_data_process as nip
def write_interaction(interaction_list,split,output_dir,tableuse,dir):
  json_split = os.path.join(dir,split+'.json')
  pkl_split = os.path.join(dir,split+'.pkl')

  ndic=dict()

  with open(json_split, 'w') as outfile:
    for interaction in interaction_list:
      json.dump(interaction, outfile, indent = 4)
      outfile.write('\n')



  new_objs = []
  _sum=0
  sumt=0

  max_num=0
  max_whe_num=0
  max_tab_num=0
  reo = 0
  change_sum = 0
  u_sum=0
  for i, obj in enumerate(interaction_list):

    new_interaction = []
    dbId = obj['database_id']

    colname = []
    bft=[]
    fas_id=[]
    for idxname in tableuse[dbId]['column_names']:
      if idxname[1] == '*':
        continue
      colname.append(idxname[1])
      fas_id.append(idxname[0])

    otb=tableuse[dbId]['table_names_original']
    ocol=tableuse[dbId]['column_names_original']

    o_tb=[]
    o_col=[]

    for y in otb:
      o_tb.append(y.lower())

    for x,y in ocol:
      if x==-1:
        continue


      o_col.append(y.lower())
    pre_tks = []
    t_id=0
    for ut in obj["interaction"]:
      sql = ut["sql"]
      _sum+=1


      usedtb = []
      usedcol = []
      for ix in range(len(tableuse[dbId]['table_names'])):
        usedtb.append('table')
      for ix in range(len(tableuse[dbId]['column_names'])):
        usedcol.append('column')
      raw_list=nip.get_raw_list(ut['utterance'])
      anstk, anstp,ansor,ans_info = getType(colname, tableuse[dbId]['table_names'], ut['utterance'],raw_list,o_tb,o_col,fas_id)
      anslt=[]
      for aor,tp in zip(ansor,anstp):
        asx=[]
        if tp=='column':
          for x,y in tableuse[dbId]['column_names']:
            if y==aor:
              asx.append(tableuse[dbId]['table_names'][x])
        anslt.append(asx)





      ut['used_table']=copy.deepcopy(usedtb)
      ut['used_column']=copy.deepcopy(usedcol)

      ut['utterance_toks'] = anstk
      ut['utterance_types'] = anstp
      ut['utterance_info'] = ans_info
      ut['utterance_or'] = ansor
      ut['utterance_lt']=anslt
      sql=sql.replace('distinct','')
      sql=sql.replace('whereclaim_status_name','where claim_status_name')
      sql = ' '.join(change2simple(sql.lower().split(' ')))

      sql = ' '.join(revise(sql.lower().split(' ')))



      v,_c=from_swap(sql.split(' '),tableuse[dbId],pre_tks)
      sql = ' '.join(v)
      change_sum+=_c

      sql = ' '.join(from_swap_for(sql.lower().split(' '),tableuse[dbId]))

      pre_tks=sql.split(' ')
      sql=sql.replace('+sum','+ sum')
      sqls = [sql]
      tok_sql_list = []
      for sql in sqls:
        results = []
        tokenized_sql = sql.split()
        tok_sql_list.append((tokenized_sql, results))

      sql1=copy.deepcopy(sql)

      ut['key_words']=nip.remove_all_label(sql1.split())
      ut['sel1_num'],ut['sel1_cols']=nip.cal_select_num(sql.split(),ocol)
      ut['whe1_num'],ut['whe1_cols']=nip.cal_where(sql.split(),ocol)
      ut['from1_num'],ut['from1_tabs']=nip.cal_from(sql.split(),otb)
      ut['dependence'] = 0
      if len(bft)!=0:
        ut['dependence'] = 1
        ok=0
        for x in ut['from1_tabs']:
          if x  in bft:
            ok=1
        if ok==0:
          sumt+=1
          ut['dependence'] = 0

      bft=ut['from1_tabs']

      max_num=max(ut['sel1_num'],max_num)
      max_whe_num=max(ut['whe1_num'],max_whe_num)
      max_tab_num=max(ut['from1_num'],max_tab_num)
      for xx in ut['key_words']:
        ndic[xx]=1

      ut["sql"] = tok_sql_list
      ut['utterance']=ut['utterance'].lower()

      new_interaction.append(ut)

    obj["interaction"] = new_interaction

    new_objs.append(obj)

  with open(pkl_split,'wb') as outfile:
    pickle.dump(new_objs, outfile)


  return



peingiemap={}

def getN_gram(toks,names,onames,cols_fa):
  col_ok = 0

  for idx,col in enumerate(names):
    tmp=onames[idx]

    col_list=col
    if len(toks) != len(col_list):
      continue
    flag1 = 1
    lsum = 0

    for x1, x2 in zip(toks, col_list):
      x1 = x1.lower()
      x2 = x2.lower()
      if x1 == x2:
        lsum+=1
        continue
      else:
        flag1 = 0


    if flag1==1  :

      col_ok = 1
      return col_ok,tmp,cols_fa[idx]
  return col_ok,'',''
def getType(col_name,tab_name,utterance,raw_list,otb,ocol,fas_id):
  tabs_raw=[]
  cols_raw=[]
  col_name_lists=[]
  for idx, col in enumerate(col_name):
    cols_raw.append(col.lower())
    col_list=nip.get_raw_list(col)
    col_name_lists.append(col_list)

  tab_name_lists = []
  for idx, tab in enumerate(tab_name):
      tabs_raw.append(tab.lower())
      tab_list = nip.get_raw_list(tab)
      tab_name_lists.append(tab_list)

  toks=utterance.split()
  nex=-1
  anstk=[]
  anstp=[]
  ansor=[]
  ans_info=[]
  for i in range(len(toks)):
    if i<=nex:
      continue
    fi=0
    for j in range(0,6):

      cptk=raw_list[i:i+6-j]
      addcptk=toks[i:i+6-j]
      col_ok,_a,_b=getN_gram(cptk,tab_name_lists,otb,tabs_raw)
      if col_ok==1:
        fi=1
        nex=i+6-j-1

        anstk.append(addcptk)

        anstp.append('table')
        ansor.append(_a)
        ans_info.append(_b)

        break

      col_ok, _a,_b = getN_gram(cptk, col_name_lists, ocol,cols_raw)
      if col_ok == 1:
          fi = 1
          nex = i + 6 - j - 1
          anstk.append(addcptk)

          anstp.append('column')
          ansor.append(_a)
          ans_info.append(_b)

          break


    if fi==0:
      anstk.append([toks[i]])
      anstp.append('te8r2ed')
      ansor.append('te8r2ed')
      ans_info.append('te8r2ed')


  return anstk,anstp,ansor,ans_info













def read_database_schema(database_schema_filename, schema_tokens, column_names, database_schemas_dict):
  with open(database_schema_filename) as f:
    database_schemas = json.load(f)

  def get_schema_tokens(table_schema):
    zidian=dict()
    column_names_surface_form = []
    column_names = []

    column_names_original = table_schema['column_names_original']

    table_names_original = table_schema['table_names_original']
    zidian['table_names']=table_schema['table_names']
    zidian['column_names']=table_schema['column_names']
    zidian['column_names_original']=table_schema['column_names_original']
    zidian['table_names_original']=table_schema['table_names_original']
    zidian['foreign_keys']=table_schema['foreign_keys']
    zidian['primary_keys']=table_schema['primary_keys']
    zidian['column_types']=table_schema['column_types']

    for i, (table_id, column_name) in enumerate(column_names_original):
      if table_id >= 0:
        table_name = table_names_original[table_id]
        column_name_surface_form = '{}.{}'.format(table_name,column_name)
      else:
        column_name_surface_form = column_name
      column_names_surface_form.append(column_name_surface_form.lower())
      column_names.append(column_name.lower())

    for table_name in table_names_original:
      column_names_surface_form.append('{}.*'.format(table_name.lower()))

    return column_names_surface_form, column_names,zidian

  tableuse=dict()
  for table_schema in database_schemas:
    database_id = table_schema['db_id']

    database_schemas_dict[database_id] = table_schema
    schema_tokens[database_id], column_names[database_id] ,tableuse[database_id]= get_schema_tokens(table_schema)

  return schema_tokens, column_names, database_schemas_dict,tableuse

class Schema:
    def __init__(self, schema, table):
      self._schema = schema
      self._table = table
      self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
      return self._schema

    @property
    def idMap(self):
      return self._idMap

    def _map(self, schema, table):
      column_names_original = table['column_names_original']
      table_names_original = table['table_names_original']
      self.table_names_original=table_names_original
      self.column_names_original=column_names_original
      for i, (tab_id, col) in enumerate(column_names_original):
        if tab_id == -1:
          idMap = {'*': i}
        else:
          key = table_names_original[tab_id].lower()
          val = col.lower()
          idMap[key + "." + val] = i

      for i, tab in enumerate(table_names_original):
        key = tab.lower()
        idMap[key] = i

      return idMap


def get_schemas_from_json(fpath):
  with open(fpath) as f:
    data = json.load(f)
  db_names = [db['db_id'] for db in data]

  tables = {}
  schemas = {}
  for db in data:
    db_id = db['db_id']
    schema = {}  # {'table': [col.lower, ..., ]} * -> __all__
    column_names_original = db['column_names_original']
    table_names_original = db['table_names_original']
    tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
    for i, tabn in enumerate(table_names_original):
      table = str(tabn.lower())
      cols = [str(col.lower()) for td, col in column_names_original if td == i]
      schema[table] = cols
    schemas[db_id] = schema

  return schemas, db_names, tables
def   read_sparc_split(split_json, interaction_list,schemas,db_names,tables):
  database_now=[]
  with open(split_json) as f:
    split_data = json.load(f)
  for interaction_data in split_data:
    db_id = interaction_data['database_id']
    final_sql = interaction_data['final']['query']
    final_utterance = interaction_data['final']['utterance']

    if db_id not in interaction_list:
      interaction_list[db_id] = []

    if 'interaction_id' in interaction_data['interaction']:
      interaction_id = interaction_data['interaction']['interaction_id']
    else:
      interaction_id = len(interaction_list[db_id])

    interaction = {}
    interaction['id'] = ''
    interaction['scenario'] = ''
    interaction['database_id'] = db_id
    interaction['interaction_id'] = interaction_id
    interaction['final'] = {}
    interaction['final']['utterance'] = final_utterance
    interaction['final']['sql'] = final_sql
    interaction['interaction'] = []
    if db_id not in database_now:
      database_now.append(db_id)

    for turn in interaction_data['interaction']:
      turn_sql = []
      print_final = False

      if 'query_toks_no_value' not in turn:
        vocab = ['UNK', 'EOS', 'select', 'from', 'value', ')', '(', 'where', '=', 'by', ',', 'count', 'group', 'order',
                 'limit', 'desc', '>', 'distinct', 'and', 'avg', 'having', '<', 'in', 'sum', 'max', 'asc', 'not', 'or',
                 'like', 'min', 'intersect', 'except', '!', 'union', 'between', '-', '+','as','on','join']

        t_t = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10']
        schema = schemas[db_id]
        table = tables[db_id]
        schema = Schema(schema, table)
        query_tok=word_tokenize(turn['query'])
        qt=[]
        for x in query_tok :
          if x==';':
            continue
          x=x.lower()
          if x in vocab or x in t_t:
            qt.append(x)
          elif '.' in x and x.split('.')[0].lower() in t_t:
            c=x.split('.')
            qt.append(c[0])
            qt.append('.')
            qt.append(c[1])
          else:
            flag=0
            for ci, co in schema.column_names_original:
              if co.lower()==x:
                qt.append(x)
                flag=1
                break
            if flag:
              continue
            for tt in schema.table_names_original:
              if tt.lower()==x:
                qt.append(x)
                flag=1
                break
            if flag:
              continue
            if qt[-1]=='value':
              continue
            qt.append('value')

        turn['query_toks_no_value']=qt

      for query_tok1 in turn['query_toks_no_value']:
        for query_tok in word_tokenize(query_tok1):
          if query_tok != '.' and '.' in query_tok:
            # invalid sql; didn't use table alias in join
            turn_sql += query_tok.replace('.',' . ').split()
            print_final = True
          else:
            turn_sql.append(query_tok)
      turn_sql = ' '.join(turn_sql)

      turn_sql = turn_sql.replace('select f_id from files as t1 join song as t2 on t1 . f_id = t2 . f_id', 'select t1 . f_id from files as t1 join song as t2 on t1 . f_id = t2 . f_id')
      turn_sql = turn_sql.replace('select name from climber mountain', 'select name from climber')
      turn_sql = turn_sql.replace('select sid from sailors as t1 join reserves as t2 on t1 . sid = t2 . sid join boats as t3 on t3 . bid = t2 . bid', 'select t1 . sid from sailors as t1 join reserves as t2 on t1 . sid = t2 . sid join boats as t3 on t3 . bid = t2 . bid')
      turn_sql = turn_sql.replace('select avg ( price ) from goods )', 'select avg ( price ) from goods')
      turn_sql = turn_sql.replace('select min ( annual_fuel_cost ) , from vehicles', 'select min ( annual_fuel_cost ) from vehicles')
      turn_sql = turn_sql.replace('select * from goods where price < ( select avg ( price ) from goods', 'select * from goods where price < ( select avg ( price ) from goods )')
      turn_sql = turn_sql.replace('select distinct id , price from goods where price < ( select avg ( price ) from goods', 'select distinct id , price from goods where price < ( select avg ( price ) from goods )')
      turn_sql = turn_sql.replace('select id from goods where price > ( select avg ( price ) from goods', 'select id from goods where price > ( select avg ( price ) from goods )')
      turn_sql = turn_sql.replace('select t1 . id , t1 . name from battle except ', 'select t1 . id , t1 . name from battle as t1 except ')




      if print_final and 'train' in split_json:
        continue

      if 'utterance_toks' in turn:
        turn_utterance = ' '.join(turn['utterance_toks']) # not lower()
      else:
        turn_utterance = turn['utterance']

      interaction['interaction'].append({'utterance': turn_utterance, 'sql': turn_sql})
    interaction_list[db_id].append(interaction)

  return interaction_list,database_now


def read_cosql(cosql_dir, interaction_list):
  schemas, db_names, tables = get_schemas_from_json('data/cosql/tables.json')
  train_json = os.path.join(cosql_dir, 'train.json')
  interaction_list,train_database = read_sparc_split(train_json, interaction_list,schemas, db_names, tables)
  dev_json = os.path.join(cosql_dir, 'dev.json')
  interaction_list,dev_database = read_sparc_split(dev_json, interaction_list,schemas, db_names, tables)

  return interaction_list,train_database,dev_database



def read_sparc(sparc_dir, interaction_list):
  schemas, db_names, tables = get_schemas_from_json('data/sparc/tables.json')


  train_json = os.path.join(sparc_dir, 'train.json')
  interaction_list,train_database = read_sparc_split(train_json, interaction_list,schemas, db_names, tables)



  dev_json = os.path.join(sparc_dir, 'dev.json')
  interaction_list,dev_database = read_sparc_split(dev_json, interaction_list,schemas, db_names, tables)

  return interaction_list,train_database,dev_database

def read_spider_split(split_json, interaction_list, database_schemas, column_names, output_vocab, schema_tokens):
  with open(split_json) as f:
    split_data = json.load(f)
  database_now=[]
  for i, ex in enumerate(split_data):
    db_id = ex['db_id']

    final_sql = []
    skip = False
    for query_tok in ex['query_toks_no_value']:
      if query_tok != '.' and '.' in query_tok:
        final_sql += query_tok.replace('.',' . ').split()
        skip = True
      else:
        final_sql.append(query_tok)
    final_sql = ' '.join(final_sql)

    if skip and 'train' in split_json:
      continue


    final_sql_parse = final_sql

    final_utterance = ' '.join(ex['question_toks'])

    if db_id not in interaction_list:
      interaction_list[db_id] = []

    interaction = {}
    interaction['id'] = ''
    interaction['scenario'] = ''
    interaction['database_id'] = db_id
    interaction['interaction_id'] = len(interaction_list[db_id])
    interaction['final'] = {}
    interaction['final']['utterance'] = final_utterance
    interaction['final']['sql'] = final_sql_parse
    interaction['interaction'] = [{'utterance': final_utterance, 'sql': final_sql_parse}]
    interaction_list[db_id].append(interaction)
    if db_id not in database_now:
      database_now.append(db_id)
  return interaction_list,database_now
def read_spider(spider_dir, database_schemas, column_names, output_vocab, schema_tokens):
  interaction_list = {}
  train_json = os.path.join(spider_dir, 'train.json')
  interaction_list,train_database = read_spider_split(train_json, interaction_list, database_schemas, column_names, output_vocab, schema_tokens)
  train_database=[]
  dev_json = os.path.join(spider_dir, 'dev.json')
  interaction_list,dev_database = read_spider_split(dev_json, interaction_list, database_schemas, column_names, output_vocab, schema_tokens)

  return interaction_list,train_database,dev_database
def read_db_split(data_dir):
  train_database = []
  with open(os.path.join(data_dir,'train_db_ids.txt')) as f:
    for line in f:
      train_database.append(line.strip())

  dev_database = []
  with open(os.path.join(data_dir,'dev_db_ids.txt')) as f:
    for line in f:
      dev_database.append(line.strip())

  return train_database, dev_database

def preprocess(dataset):
  # Validate output_vocab
  output_vocab = ['_UNK', '_EOS', '.', 't1', 't2', '=', 'select', 'from', 'as', 'value', 'join', 'on', ')', '(', 'where', 't3', 'by', ',', 'count', 'group', 'order', 'distinct', 't4', 'and', 'limit', 'desc', '>', 'avg', 'having', 'max', 'in', '<', 'sum', 't5', 'intersect', 'not', 'min', 'except', 'or', 'asc', 'like', '!', 'union', 'between', 't6', '-', 't7', '+', '/']

  if dataset == 'spider':
    spider_dir = 'data/spider'
    database_schema_filename = 'data/spider/tables.json'
    output_dir = 'data/spider_data'
  elif dataset == 'sparc':
    sparc_dir = 'data/sparc/'
    database_schema_filename = 'data/sparc/tables.json'
    output_dir = 'data/sparc_data'
  elif dataset=='cosql':
    cosql_dir = 'data/cosql/'
    database_schema_filename = 'data/cosql/tables.json'
    output_dir = 'data/cosql_data'

  train_database=[]
  dev_database=[]
  if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
  os.mkdir(output_dir)

  schema_tokens = {}
  column_names = {}
  database_schemas = {}

  print('Reading spider database schema file')
  schema_tokens, column_names, database_schemas,tableuse = read_database_schema(database_schema_filename, schema_tokens, column_names, database_schemas)

  output_database_schema_filename = os.path.join(output_dir, 'tables.json')
  with open(output_database_schema_filename, 'w') as outfile:
    json.dump([v for k,v in database_schemas.items()], outfile, indent=4)






  interaction_list = {}



  if dataset == 'spider':
    interaction_list,train_database,dev_database = read_spider(spider_dir, database_schemas, column_names, output_vocab, schema_tokens)
  elif dataset == 'sparc':
    interaction_list,train_database,dev_database = read_sparc(sparc_dir, interaction_list)
  elif dataset=='cosql':
    interaction_list,train_database,dev_database = read_cosql(cosql_dir, interaction_list)


  train_interaction = []

  dev_interaction = []
  for database_id in interaction_list:
    if database_id not in dev_database:
      train_interaction += interaction_list[database_id]

  for database_id in dev_database:
    dev_interaction += interaction_list[database_id]

  print('train interaction: ', len(train_interaction))
  print('dev interaction: ', len(dev_interaction))
  write_interaction(train_interaction, 'train', output_dir,tableuse,output_dir)

  write_interaction(dev_interaction, 'dev', output_dir,tableuse,output_dir)
  return
CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except','end')
SQL_OPS = ('intersect', 'union', 'except')

def list_swap(sql_toks,lr1,lr2):
    tk1=copy.deepcopy(sql_toks[lr1[0]:lr1[1]])
    tk2=copy.deepcopy(sql_toks[lr2[0]:lr2[1]])
    sql_toks[lr1[0]:lr1[0]+len(tk2)]=tk2
    sql_toks[lr1[0]+len(tk2):lr1[0]+len(tk2)+len(tk1)]=tk1
    return sql_toks

def revise(sqltok):

  sqltok.append('end')
  s=0
  for x in sqltok:
    if x in SQL_OPS:
      s+=1

  t=(sqltok.count('from'))
  t1=(sqltok.count('select'))

  s_list=[]
  l = -1
  for idx,x in enumerate(sqltok):

    if l != -1 and x in CLAUSE_KEYWORDS:
      s_list.append([l, idx])
      l = -1
    if x=='select':
      l=idx

  f_list = []
  l = -1
  for idx, x in enumerate(sqltok):

    if l != -1 and (x in CLAUSE_KEYWORDS or x==')'):
      f_list.append([l, idx])
      l = -1
    if x == 'from':
      l = idx

  for lr1,lr2 in zip(s_list,f_list):
    sqltok=list_swap(sqltok,lr1,lr2)



  return sqltok[0:-1]



def foreign_judge(sqltok):

  sqltok.append('end')
  l=-1
  pair=[]
  t_d=dict()
  e=-1
  for idx, x in enumerate(sqltok):

    if l != -1 and (x in CLAUSE_KEYWORDS or x==')'):
      for p in pair:
        tl=p[0]
        tr=p[1]
      t_d=dict()
      pair=[]
      l = -1
    if x == 'from':
      l = idx

    if l!=-1 and x=='on':
      tl = sqltok[idx + 1]
      tr = sqltok[idx + 5]
      if tr=='class_senator_vote' or tr=='secretary_vote' or tr=='vice_president_vote' or tr=='president_vote':
        tr='t2'
      if tl=='friend_id':
        tl='t1'
        tr='t2'
      if tl == 'student_id':
        tl = 't2'
        tr = 't3'
      pair.append([tl,tr])
    if l!=-1 and x=='as':
      t_d[sqltok[idx+1]]=sqltok[idx-1]


def judge_from(sqltok,ocol):
  tb=dict()
  sqltok=changeerr(sqltok)
  oc=[]
  for x,y in ocol:
    oc.append(y.lower())
  tt=['t1','t2','t3','t4','t5']
  sqltok.append('end')
  l=-1
  for idx,tk in enumerate(sqltok):
    if l != -1 and (tk in CLAUSE_KEYWORDS or tk==')'):
      l=-1
    if tk == 'from':
      l = idx

    if l!=-1 and tk in tt and sqltok[idx-1]=='as':
      tb[tk]=sqltok[idx-2]
    if l!=-1 and tk=='.':
      if sqltok[idx-1] not in tb:

        sqltok[idx - 1]='t2'


  l = -1
  for idx, tk in enumerate(sqltok):
    if l != -1 and (tk in CLAUSE_KEYWORDS or tk == ')'):
      l = -1
    if tk == 'from':
      l = idx

    if l != -1 and tk in tt and sqltok[idx - 1] == 'as':
      tb[tk] = sqltok[idx - 2]
    if l != -1 and tk == '.':
      if sqltok[idx - 1] not in tb:

        sqltok[idx - 1] = 't2'

  new_sql=[]
  for idx,tk in enumerate(sqltok):
    new_sql.append(tk)
    if tk =='=' and sqltok[idx+1] in oc:
      new_sql.append('t2')
  sqltok=new_sql

  return sqltok[:-1]

def change2simple(sql):
  nsql=[]
  e=-1
  for idx, x in enumerate(sql):

    if idx<=e:
      continue
    if x=='on':
      e=idx+7
      continue
    nsql.append(x)
  nsql=remove_t1_t2(nsql)
  return nsql


def remove_t1_t2(sql):
  ri=['t1','t2','t3','t4','t5','t6','t7','t8','t9','.','as','join']
  ans=[]
  for x in sql:
    if x in ri:
      continue
    ans.append(x)
  return ans


fgf = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except','end','having')

xrc=['avg','min','max','sum','count']
def update_data(sql,cols):

  sql=sql.replace('count ( * )','whole')
  sql=sql.replace('distinct','')
  sql_tks=sql.split()
  sql_tks.append('end')

  all_col=[]

  bbf_fgf=''
  for idx,tk in enumerate(sql_tks):


    if tk in fgf :

      bbf_fgf=tk
      all_col=[]
    if tk in cols:
      all_col.append(tk)



def word_swap_select(pre_dict_cols,sql_toks,ocols):
  now_q = []
  sp = -1
  cols = pre_dict_cols
  new_tks = []
  next_col = []
  for idx, tk in enumerate(sql_toks):
    if idx <= sp:
      continue
    if tk == 'select':
      sp = idx
      new_tks.append('select')
      for tk_1 in sql_toks[idx + 1:]:
        sp += 1
        if tk_1 in ocols:
          if sql_toks[sp-1]=='(':
            now_q.append(' '.join(sql_toks[sp-2:sp+2]))
          else:
            now_q.append(tk_1)
        if tk_1 == 'from':
          for y in cols:
            if y in now_q:
              ny=y.split()

              for yy in ny:
                new_tks.append(yy)

              next_col.append(y)
          for y in now_q:
            if y in cols:
              continue
            ny = y.split()
            for yy in ny:
              new_tks.append(yy)
            next_col.append(y)
          now_q = []
          break
      new_tks.append('from')
    else:
      new_tks.append(tk)

  return new_tks, next_col

def from_swap_for(sql_toks,input_schema):
  foreign_keys = input_schema['foreign_keys']
  ocols_tb = [x[0] for x in input_schema['column_names_original']]
  allTable =[x.lower() for x in input_schema['table_names_original']]
  CLAUSE_KEYWORDS1 = ('select',  'where', 'end')
  sql_toks.append('end')
  flag=0
  ntb=[]
  ntk=[]
  for tk in sql_toks:
    if tk=='from':
      flag=1
    if tk==CLAUSE_KEYWORDS1 and flag==1:
      if len(ntb)>0:
        vis_tab=[ntb[0]]

        while True:
          fo_tb=[]
          for k1, k2 in foreign_keys:
            tbid1 = ocols_tb[k1]
            tbid2 = ocols_tb[k2]
            tb_1 = allTable[tbid1]
            tb_2 = allTable[tbid2]
            if tb_1 in vis_tab or tb_2 in vis_tab:
              fo_tb.append(tb_1)
              fo_tb.append(tb_2)
          t_ok=0
          for x in ntb:
            if x in vis_tab:
              continue
            if x in fo_tb:
              vis_tab.append(x)
              t_ok=1
          if t_ok==0:
            break
          for tt in vis_tab:
            ntk.append(tt)




      ntb=[]
      flag=0
    if flag==1 and tk in ocols_tb:
      ntb.append(tk)
    else:
      ntk.append(tk)


  return ntk[:-1]

def from_swap(sql_tks,input_schema,pre_tks):
  if len(pre_tks)==0:
    return sql_tks,0
  tabs = [x.lower() for x in input_schema['table_names_original']]
  n_f = ('select',  'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'end', 'having')
  s1=[]
  s2=[]
  flag=0
  n_tks=[]
  sql_tks.append('end')
  for x in sql_tks:

    if flag==1 and x in n_f:
      flag=0
      for tt in s1:
        n_tks.append(tt)
      for tt in s2:
        n_tks.append(tt)
      s1=[]
      s2=[]

    if x in tabs and flag==1:
      if x in pre_tks:
        s1.append(x)
      else:
        s2.append(x)
    if flag == 0:
      n_tks.append(x)

    if x == 'from':
      flag = 1
  _=0
  if n_tks!=sql_tks:

    _+=1
  return n_tks[:-1],_

def only_use_from(sql_toks,txt_split_w):

  n_f = ('select', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'end', 'having')
  xxx = ['intersect', 'union', 'except']
  n_sql_toks = []
  u_sum = 0
  flag = 1
  _from_ci = 0

  for x in sql_toks:
    if x in n_f:
      flag = 0

    if x == 'from':

      if len(n_sql_toks) > 0 and n_sql_toks[-1] not in xxx:
        # break
        n_sql_toks.append('intersect')
        #
      flag = 1
      _from_ci += 1
    if flag == 1 :
      if x=='from':
        if len(n_sql_toks)==0:
          n_sql_toks.append(x)

      else:
        n_sql_toks.append(x)

    if x in xxx:
      n_sql_toks.append('intersect')

  return n_sql_toks, u_sum, _from_ci

def getfa_dic(ans_info,ans_type,tb_dic,t_id):

  t_ys = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10']
  c_ys = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']


  for idx,x in  enumerate(ans_type):
    if x=='table' or x=='column':
      if ans_info[idx] not in tb_dic:
        tb_dic[ans_info[idx]]=t_ys[t_id]
        t_id+=1

  new_type=[]

  for idx,x in enumerate(ans_type):
    if x=='table' or x=='column':
      if x=='table':
        new_type.append(tb_dic[ans_info[idx]])
      else:
        n_id=t_ys.index(tb_dic[ans_info[idx]])
        new_type.append(c_ys[n_id])

    else:
      new_type.append(x)
  return new_type,tb_dic,t_id




import argparse
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='sparc')

  args = parser.parse_args()

  preprocess(args.dataset)