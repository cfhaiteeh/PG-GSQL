This is a PyTorch implementation of the PG-GSQL in our COLING 2020 paper "[PG-GSQL: Pointer-Generator Network with Guide Decoding for Cross-Domain Context-Dependent Text-to-SQL Generation](https://www.aclweb.org/anthology/2020.coling-main.33/)".



### Run SParC experiment

####  Requirements
You can refer to `requirements.txt` 

#### Pretrained BERT
You need download the pretrained BERT from [there](https://drive.google.com/file/d/1-tFqErsoMZYrPdiyozFyxXGhzvgbP4S2/view?usp=sharing) and put them in `/model/bert/data/`

####  Dataset
##### Two options are available
1) You can get dataset from  `https://github.com/taoyds/sparc` and put them in the `/data/` folder
, then run the `python3 preprocess_data.py --dataset sparc` to preprocess the data.
2) You can use our preprocessed data and  download the database from [there](https://drive.google.com/file/d/1mHeGXXEj2BIo59TzPqMT5JrL2lxGdQIn/view?usp=sharing).
#### Run model
Train `sh ./run_sparc_pg_gsql.sh`

Eval `sh ./eva_att.sh`

#### Reproduce our model 

1) You need download our trained model from [there](https://drive.google.com/file/d/18BwkTr2F7OeoL-R0X-Nl2qSvVs11QMFM/view?usp=sharing) and put it in `/sparc_pg_gsql_paper_save/`
2) change the `dir` in `evaluate_g.py` and run `sh ./eva_att.sh`
3) You can get the performance on the dev set.


<table>
  <tr>
    <td></td>
    <td>question matching</td>
    <td>interaction matching</td>
  </tr>
  <tr>
    <td>PG-GSQL</td>
    <td>53.1</td>
    <td>34.7</td>
  </tr>
</table>

#### Reference

`https://github.com/taoyds/sparc`

`https://github.com/ryanzhumich/editsql`

`https://github.com/lil-lab/atis`

#### Our paper bibtex

`@inproceedings{wang-etal-2020-pg,
    title = "{PG}-{GSQL}: Pointer-Generator Network with Guide Decoding for Cross-Domain Context-Dependent Text-to-{SQL} Generation",
    author = "Wang, Huajie  and
      Li, Mei  and
      Chen, Lei",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.33",
    doi = "10.18653/v1/2020.coling-main.33",
    pages = "370--380",
}`