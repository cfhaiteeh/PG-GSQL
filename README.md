This is a PyTorch implementation of the PG-GSQL in our COLING 2020 paper "[PG-GSQL: Pointer-Generator Network with Guide Decoding for Cross-Domain Context-Dependent Text-to-SQL Generation](https://www.aclweb.org/anthology/2020.coling-main.33/)".



### Run SParC experiment

####  Requirements
You can refer to `requirements.txt` 

####  Dataset
##### Two options are available
1) You can get dataset from  `https://github.com/taoyds/sparc` and put them in the `/data/` folder
, then run the `python3 preprocess_data.py --dataset sparc` to preprocess the data.
2) You can use our preprocessed data and  download the database from ''.
#### Run model