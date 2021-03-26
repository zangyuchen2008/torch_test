from transformers import AutoModel,AutoConfig,AutoTokenizer
from torch import nn
import torch
import pandas as pd
from torch.utils.data import  dataloader
from transformers import AdamW
from sklearn.metrics import accuracy_score

# config = AutoConfig.from_pretrained('model/distilbert/config.json')
# model = AutoModel.from_config(config)
# tokenizer = AutoTokenizer.from_pretrained('/data/yuchen/projects/test/transformers_test/model/distilbert')
# inputs = tokenizer('hello world',return_tensors='pt')
# model(**inputs)



def load_disdbert():
    config = AutoConfig.from_pretrained('/data/yuchen/projects/transformers_test/model/distilbert/config.json')
    model = AutoModel.from_config(config)
    # tokenizer = AutoTokenizer.from_pretrained('/data/yuchen/projects/test/transformers_test/model/distilbert')
    # inputs = tokenizer('hello world',return_tensors='pt')
    return model

class Classification(nn.Module):
    def __init__(self):
        super(Classification,self).__init__()
        self.distilbert = load_disdbert()
        self.ffn = nn.Linear(768,2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,inputs):
        cls = self.distilbert(**inputs)['last_hidden_state'][:,0,:]
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
        pro = nn.functional.softmax(logits,dim=1)
        pre = pro.argmax(dim=1)
        pre = pre.detach().cpu().numpy()
        accuracy_score(label,pre)
        valid_result.append(accuracy_score(label,pre))   
    modle.train()
    return sum(valid_result) / len(valid_result)


if __name__ == "__main__":
    BATCH_SIZE=8
    imdb = pd.read_csv('/data/yuchen/projects/transformers_test/data/IMDB Dataset.csv')
    imdb.sentiment = imdb.sentiment.apply(lambda x: 0 if x == 'positive' else 1)#.astype(float)
    imdb.review = imdb.review.astype(str)
    data = list(zip(imdb.review.to_list(),imdb.sentiment.to_list()))
    # dl = dataloader.DataLoader(data,batch_size=BATCH_SIZE,shuffle=True)
    train_dl = dataloader.DataLoader(data[:-1000],batch_size=BATCH_SIZE,shuffle=True)
    valid_dl = dataloader.DataLoader(data[-1000:],batch_size=BATCH_SIZE,shuffle=True)

    torch.cuda.set_device(1)
    modle = Classification().cuda()
    tokenizer = AutoTokenizer.from_pretrained('/data/yuchen/projects/transformers_test/model/distilbert')
    
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
    for index,batch in enumerate(train_dl):
        train_data = tokenizer(list(batch[0]),return_tensors='pt',padding='max_length',max_length=512,truncation=True).to(torch.cuda.current_device())
        target =  batch[1].to(torch.cuda.current_device())
        score = modle(train_data)
        loss =  nn.functional.cross_entropy(score,target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if index % 100 ==0: 
            acc  = evaluate(valid_dl,modle,tokenizer)
            print('current loss is: {}'.format(loss))
            print('evaluation acc: {}'.format(acc))
