# ref : http://github.com/dbiir/UER-py

import os
from transformers import BertTokenizer, BertModel

# load model. i.e. uer/chinese_roberta_L-2_H-512 
model_class_prefix = 'uer'
model_name = 'chinese_roberta_L-2_H-512'
model_path = model_class_prefix + '/' + model_name

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


model_to_save = model.module if hasattr(model, 'module') else model
if not os.path.exists('./pretrain_model/'+model_path):
    print("no such file:", './pretrain_model/'+model_path)
    os.makedirs('./pretrain_model/'+model_path)
    print("create new path:",os.path.exists('./pretrain_model/'+model_path))


# pretrain_save
model.save_pretrained("./pretrain_model/"+model_path)
tokenizer.save_pretrained("./pretrain_model/"+model_path)
