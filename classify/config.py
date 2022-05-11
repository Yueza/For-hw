# data path
data_dir = '../data'
subj = '初中语文'
bert_path = '/home/yuezhiang/pretrain_model/chinese_roberta_L-12_H-768'


max_len = 512
epoch = 30
lr = 1e-5
batch_size = 24
dropout_rate = 0.1

logging_file_name = f'./result/log/{subj}_epoch_{epoch}_batch_{batch_size}.json'
save_path = f'./result/model'