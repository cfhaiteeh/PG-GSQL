t_label=['t1','t2','t3','t4','t5','t6','t7','.']
def remove_t_label(sql_toks):
    new_toks=[]
    for tk in sql_toks:
        if tk in t_label:
            continue
        new_toks.append(tk)

    return new_toks


CLAUSE_KEYWORDS = ('select','having', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except','desc','asc')
SQL_OPS = ('intersect', 'union', 'except')
def remove_all_label(sql_toks):
    new_toks=[]
    for idx,tk in enumerate(sql_toks):
        if tk in CLAUSE_KEYWORDS:
           new_toks.append(tk)
        elif tk=='by' and (sql_toks[idx-1]=='group' or sql_toks[idx-1]=='order'):
            new_toks.append(tk)
    return new_toks

def cal_select_num(sql_toks,column_names):

    nc=[ tk[1].lower() for tk in column_names]
    num=0
    flag=0
    sel1_cols=[]
    for tk in sql_toks:
        tk=tk.lower()

        if flag==1 and tk in CLAUSE_KEYWORDS:
            break
        if tk =='select' and flag==0:
            flag=1
        if flag==1 and (tk in nc or tk =='*'):
            num+=1
            sel1_cols.append(tk)
    return num,sel1_cols

def cal_where(sql_toks,column_names):
    nc = [tk[1].lower() for tk in column_names]
    num = 0
    flag = 0
    whe1_cols = []
    for tk in sql_toks:
        tk = tk.lower()
    
        if flag == 1 and tk in CLAUSE_KEYWORDS:
            break
        if tk == 'where' and flag == 0:
            flag = 1
        if flag == 1 and (tk in nc or tk == '*'):
            num += 1
            whe1_cols.append(tk)
    return num, whe1_cols

def cal_from(sql_toks,table_names):
   
    nc = [tk.lower() for tk in table_names]

    num = 0
    flag = 0
    from_tabs = []
    for tk in sql_toks:
        tk = tk.lower()
    
        if flag == 1 and tk in CLAUSE_KEYWORDS:
            break
        if tk == 'from' and flag == 0:
            flag = 1
        if flag == 1 and (tk in nc ):
            num += 1
            from_tabs.append(tk)
    return num, from_tabs