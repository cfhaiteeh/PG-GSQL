CLAUSE_KEYWORDS = ['select', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except','on','having','EOS']

def key_gen(sql_toks,input_schema):
    if 'intersect' in sql_toks:
        inid=sql_toks.index('intersect')
        if sql_toks[inid-2]=='>' or sql_toks[inid-2]=='<':
            snm=sql_toks.count(sql_toks[inid-2])
            if snm==2:
                if sql_toks[inid-2]=='>':
                    sql_toks[inid-2]='<'
                else:
                    sql_toks[inid-2]='>'
    if 'and' in sql_toks:
        e=-1
        n_sql=[]
        for idx,x in enumerate(sql_toks):
            if idx<=e:
                continue
            if x=='and':
                cmp1=[]
                for y in sql_toks[idx+1:]:
                    if y in CLAUSE_KEYWORDS:
                        break
                    cmp1.append(y)
                if cmp1==sql_toks[idx-len(cmp1):idx]:
                    e=idx+len(cmp1)
                else:
                    n_sql.append(x)
            else:
                n_sql.append(x)
        sql_toks=n_sql

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

import numpy as np
def common_error_revise(primary_keys,argmax_token,argmax_index,sequence,input_sequence,previous_queries,probabilities,distribution_map,allCol):
    if argmax_token=='name' and sequence[-1]=='select' and ('named' not in input_sequence and 'name' not in input_sequence and 'names' not in input_sequence and ( len(previous_queries)==0 or 'name' not in previous_queries[-1])):
        ncs=0
        for idx,ov in enumerate(allCol):
            if ov.lower() == argmax_token and idx in primary_keys:
                ncs=1
                break
        
        if sequence[-1]!='where' and ncs==0:
            argmax_token='*'
    if argmax_token in allCol:
        if (sequence[-1]==',' and sequence[-2]==argmax_token)  :
            probabilities[argmax_index] = 0.
            argmax_index = int(np.argmax(probabilities))
            argmax_token = distribution_map[argmax_index]
    if argmax_token=='*' and sequence[-1]=='where':
            probabilities[argmax_index] = 0.
            argmax_index = int(np.argmax(probabilities))
            argmax_token = distribution_map[argmax_index]
    return argmax_token,argmax_index


def judge_is_table(argmax_token,allTable,flag,table_id,vis_tab):
    if(argmax_token=='from'):
        table_id=[]
        vis_tab=[]
        flag=1
    if flag==1:
        if argmax_token in allTable:
            table_id.append(allTable.index(argmax_token))
            vis_tab.append(argmax_token)
    if(flag==1 and argmax_token in CLAUSE_KEYWORDS):
        flag=0
    return flag,table_id,vis_tab
                      