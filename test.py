from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import BertTokenizer
from torch.utils.data import  dataloader
from transformers import AdamW
import torch
import pandas as pd
from torch.nn import functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
torch.cuda.set_device(1)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased').cuda()
model.train()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)


BATCH_SIZE=6
imdb = pd.read_csv('/data/yuchen/projects/transformers_test/data/IMDB Dataset.csv')
imdb.sentiment = imdb.sentiment.apply(lambda x: 0 if x == 'positive' else 1)#.astype(float)
imdb.review = imdb.review.astype(str)
data = list(zip(imdb.review.to_list(),imdb.sentiment.to_list()))
dl = dataloader.DataLoader(data,batch_size=BATCH_SIZE,shuffle=True)

for index,batch in enumerate(dl):
    encoding = tokenizer(batch[0], return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].cuda()
    attention_mask = encoding['attention_mask'].cuda()
    labels = batch[1].cuda()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    # loss = outputs.loss
    loss = F.cross_entropy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    if index % 50 == 0: print(loss)