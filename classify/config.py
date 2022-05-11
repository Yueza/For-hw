# data path
data_dir = '../data'
subj = '初中语文.json'
bert_path = '/mnt/yza/pretrain_model/chinese_roberta_L-12_H-768'


max_len = 256
epoch = 10
lr = 0.001
batch_size = 32
dropout_rate = 0.1

logging_file_name = f'./result/log/{subj}_epoch_{epoch}_batch_{batch_size}.json'
save_path = f'./result/model'