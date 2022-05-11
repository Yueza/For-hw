import logging
import json

def get_logger(filename, verbosity=1, name=None, mode="a"):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

#读写文件
def LoadJson(file_path):
	result = []
	with open(file_path, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line != '': result.append(json.loads(line))
	return result

def LoadList(file_path):
	with open(file_path, encoding="utf-8") as fin:
		result = list(ll for ll in fin.read().split('\n') if ll != "")
	return result

def SaveJson(data, file_path): 
    data_str = [json.dumps(x, ensure_ascii=False) for x in data]
    with open(file_path, "w", encoding = "utf-8") as fout:
        for k in data_str:
            fout.write(str(k) + "\n")