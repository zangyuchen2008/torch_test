from os import EX_NOPERM
from transformers import AutoModel,AutoConfig,AutoTokenizer
from torch import nn
import torch
import pandas as pd
from torch.utils.data import  dataloader
from transformers import AdamW
from sklearn.metrics import accuracy_score
import pickle
from tqdm import tqdm
import os
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s",
    handlers=[
        logging.FileHandler("train_0617_1455.log"),
        logging.StreamHandler()
    ]
)
# training test data comes from https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# base bert model form hugging face
torch.cuda.empty_cache()
def load_dbert():
    # you can load directly via auto downloading
    config = AutoConfig.from_pretrained('/home/zangyuchen/bst_agent/data/model/bert/config.json')
    model = AutoModel.from_config(config)
    # tokenizer = AutoTokenizer.from_pretrained('/data/yuchen/projects/test/transformers_test/model/distilbert')
    # inputs = tokenizer('hello world',return_tensors='pt')
    return model

class Classification(nn.Module):
    def __init__(self):
        super(Classification,self).__init__()
        self.bert = load_dbert()
        self.ffn = nn.Linear(768,3)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,inputs):
        cls = self.bert(**inputs)['last_hidden_state'][:,0,:]
        cls =  self.ffn(cls)
        # cls = self.softmax(cls)
        return cls

def loss(score,label):
    loss = nn.functional.cross_entropy(score,label)
    return loss
    
def evaluate(eval_data,model,tokenizer):
    modle.eval()
    valid_result = []
    for valid_data, label in eval_data:
        inputs = tokenizer(list(valid_data),return_tensors='pt',padding='max_length',max_length=512,truncation=True).to(torch.cuda.current_device())
        logits = model(inputs)
        del inputs
        torch.cuda.empty_cache()
        pro = nn.functional.softmax(logits,dim=1)
        pre = pro.argmax(dim=1)
        pre = pre.detach().cpu().numpy()
        accuracy_score(label,pre)
        valid_result.append(accuracy_score(label,pre))   
    modle.train()
    return sum(valid_result) / len(valid_result)

# data labe : wow 0, ed 1, convai 2
if __name__ == "__main__":
    EPOCH =1
    BATCH_SIZE=4
    version1_context_labels = pickle.load(open('/home/zangyuchen/bst_agent/data/version1_context_labels.pkl','rb'))
    datas = [version1_context_labels]
    data_description= ['version1_context_labels']
    for data,desc in list(zip(datas,data_description))[-1:]:
        logging.info(desc + " started>>>>>>>>")
        # train = data[:7000]
        train = data[:350000]  #[:10000]#list(dict(data[:300000][:10000]).items())
        valid = data[350000:355000]   #list(dict(data[300000:][:200]).items())
        test = data[370000:]
        train_dl = dataloader.DataLoader(train,batch_size=BATCH_SIZE,shuffle=True)
        valid_dl = dataloader.DataLoader(valid,batch_size=BATCH_SIZE,shuffle=True)

        torch.cuda.set_device(0)
        modle = Classification().cuda()
        # modle.load_state_dict(torch.load('/home/zangyuchen/bst_agent/checkpoint/checkpoint_0616_0.96/checkpoint75000/model.bin'))
        tokenizer = AutoTokenizer.from_pretrained('/home/zangyuchen/bst_agent/data/model/bert')
        
        param_optimizer = list(modle.named_parameters())
        optimizer_params = {'lr': 2e-5} 
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
        optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)    
        
        modle.zero_grad()
        modle.train()
        valid_accs = []
        # loss_weight = torch.Tensor([1.0,1.0,1.0,0.5,1.0]).to(torch.cuda.current_device())
        # loss_weight = torch.Tensor([1.0,1.0,0.3]).to(torch.cuda.current_device())
        for epoch in range(EPOCH):
            logging.info("this is epoch {}".format(epoch))
            losses = []
            for index,batch in tqdm(enumerate(train_dl)):
                train_data = tokenizer(list(batch[0]),return_tensors='pt',padding='max_length',max_length=512,truncation=True).to(torch.cuda.current_device())
                target =  batch[1].to(torch.cuda.current_device())
                score = modle(train_data)
                # loss =  nn.functional.cross_entropy(score,target,weight=loss_weight)
                loss =  nn.functional.cross_entropy(score,target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss)
                eval_index = 5000
                save_index = 25000
                if index % eval_index ==0: 
                    # train_acc  = evaluate(train_dl,modle,tokenizer)
                    valid_acc  = evaluate(valid_dl,modle,tokenizer)
                    valid_accs.append(valid_acc)
                    logging.info('train loss is: {}'.format(sum(losses)/len(losses)))
                    logging.info('current valid acc: {}'.format(valid_acc))
                    logging.info('current valid acc at checkpoint: {}'.format(index))
                    logging.info('best valid acc: {}'.format(max(valid_accs)))
                    logging.info('best valid acc at checkpoint: {}'.format((valid_accs.index(max(valid_accs)))*eval_index))
                    if (valid_acc >= max(valid_accs)) or (index % save_index == 0):
                        logging.info('saving model')
                        path = '/home/zangyuchen/bst_agent/checkpoint/checkpoint0617_1/checkpoint'+ str(index)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        torch.save(modle.state_dict(), path+ '/model.bin')
                        logging.info('saving finished')     
                    logging.info('****next ' + str(eval_index) + ' batch****') 
                del train_data
                torch.cuda.empty_cache()

        del modle
        torch.cuda.empty_cache()
