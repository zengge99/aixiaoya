import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os
import re
import random
import numpy as np
import glob
from transformers import AutoTokenizer, AutoModel, logging

# --- 配置区 ---
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 核心配置
NUM_THREADS = 4       
BATCH_SIZE = 32       
LR = 5e-5             
EPOCHS = 10           
MAX_LEN = 128         

# === 模型路径配置 (关键修改) ===
# 基础预训练模型名称 (网上下载的源头)
BERT_HF_NAME = "prajjwal1/bert-tiny" 
# 本地保存的基础模型文件夹 (下载一次后就存这儿)
BERT_LOCAL_FOLDER = "bert_base_local"
# 我们训练好的权重文件
MODEL_WEIGHTS_PATH = "movie_model_bert.pth"

DATA_FILE_PATTERN = "train_data*.txt"
SEED = 42
THRESHOLD = 0.5       

torch.set_num_threads(NUM_THREADS)

# --- 工具类 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TextUtils:
    @staticmethod
    def preprocess_for_bert(text):
        if not text: return ""
        text = re.sub(r'[._]', ' ', text)
        chars = r'a-zA-Z0-9\u4e00-\u9fa5'
        puncts = r'\[\]\(\)\{\}\-\'\"\:!&' 
        cn_puncts = r'【】（）《》：'
        pattern = f'[^{chars}{puncts}{cn_puncts}\s]'
        text = re.sub(pattern, ' ', text)
        return text

    @staticmethod
    def cleanup_result(text):
        if not text: return ""
        text = text.strip(" .-_[]()/\\")
        text = re.sub(r'\s+', ' ', text)
        return text

# --- 自动下载并缓存 BERT 的逻辑 ---
def get_bert_path():
    """
    检查本地是否有 BERT 基础模型。
    如果没有，从 HuggingFace 下载并保存到本地。
    返回有效的模型路径。
    """
    if os.path.exists(BERT_LOCAL_FOLDER) and os.listdir(BERT_LOCAL_FOLDER):
        # 文件夹存在且不为空，说明已经下载过了
        return BERT_LOCAL_FOLDER
    else:
        print(f"检测到本地无基础模型，正在从 {BERT_HF_NAME} 下载...")
        try:
            # 下载 Tokenizer 和 Model
            tokenizer = AutoTokenizer.from_pretrained(BERT_HF_NAME)
            model = AutoModel.from_pretrained(BERT_HF_NAME)
            
            # 保存到本地文件夹
            os.makedirs(BERT_LOCAL_FOLDER, exist_ok=True)
            tokenizer.save_pretrained(BERT_LOCAL_FOLDER)
            model.save_pretrained(BERT_LOCAL_FOLDER)
            
            print(f"基础模型已保存到: {BERT_LOCAL_FOLDER} (以后不再联网下载)")
            return BERT_LOCAL_FOLDER
        except Exception as e:
            print(f"下载失败: {e}")
            sys.exit(1)

# --- 模型定义 ---
class BertExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # 这里传入的是本地路径
        self.bert = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size 
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        probs = self.classifier(sequence_output).squeeze(-1)
        return probs

# --- 数据集 ---
class MovieDataset(Dataset):
    def __init__(self, lines, tokenizer, max_len=MAX_LEN):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        for line in lines:
            line = line.strip()
            if '#' not in line: continue
            input_path, target_name = line.rsplit('#', 1)
            target_name = target_name.strip()
            if not target_name: continue
            escaped = re.escape(target_name).replace(r'\ ', r'.*')
            if re.search(escaped, input_path, re.IGNORECASE):
                self.samples.append((input_path, target_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_name = self.samples[idx]
        escaped_target = re.escape(target_name)
        pattern = escaped_target.replace(r'\ ', r'.+') 
        matches = list(re.finditer(pattern, input_path, re.IGNORECASE))
        if matches:
            match = matches[-1]
            start_char, end_char = match.start(), match.end()
        else:
            start_char, end_char = -1, -1

        text_for_bert = TextUtils.preprocess_for_bert(input_path)
        encoding = self.tokenizer(
            text_for_bert,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0)

        labels = torch.zeros(self.max_len, dtype=torch.float)
        if start_char != -1:
            for i, (start, end) in enumerate(offset_mapping):
                if attention_mask[i] == 0 or start == end: continue 
                overlap_start = max(start, start_char)
                overlap_end = min(end, end_char)
                overlap = max(0, overlap_end - overlap_start)
                token_len = end - start
                if token_len > 0 and (overlap / token_len > 0.4):
                    labels[i] = 1.0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# --- 损失函数 ---
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=6.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, mask):
        loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        weights = targets * self.pos_weight + (1 - targets)
        loss = loss * weights * mask
        return loss.sum() / (mask.sum() + 1e-6)

# --- 训练流程 ---
def run_train():
    set_seed(SEED)
    
    # 1. 获取 BERT 基础模型路径 (只下载一次)
    bert_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

    # 读取数据
    data_files = glob.glob(DATA_FILE_PATTERN)
    if not data_files:
        print("未找到训练数据 train_data*.txt"); return
        
    all_lines = []
    for f in data_files:
        with open(f, 'r', encoding='utf-8') as file:
            all_lines.extend(file.readlines())
    
    random.shuffle(all_lines)
    split = int(len(all_lines) * 0.95)
    train_ds = MovieDataset(all_lines[:split], tokenizer)
    val_ds = MovieDataset(all_lines[split:], tokenizer)
    
    print(f"训练集: {len(train_ds)} | 验证集: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=min(2, NUM_THREADS))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 从本地加载模型结构
    model = BertExtractor(bert_path)
    
    # 如果有旧的训练权重，加载继续练
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print("加载已有微调权重...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
        
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = WeightedBCELoss(pos_weight=6.0)
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            preds = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(preds, batch['labels'], batch['attention_mask'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                preds = model(batch['input_ids'], batch['attention_mask'])
                loss = criterion(preds, batch['labels'], batch['attention_mask'])
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f" >> Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            print(" >> 模型已保存 (Best)")

# --- 预测流程 ---
def predict_single(raw_path, model, tokenizer):
    if not raw_path.strip() or raw_path.startswith('#'): return
    raw_path = raw_path.strip()

    text_for_bert = TextUtils.preprocess_for_bert(raw_path)
    inputs = tokenizer(
        text_for_bert,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LEN
    )
    
    with torch.no_grad():
        probs = model(inputs['input_ids'], inputs['attention_mask'])[0].numpy()
        
    offset_mapping = inputs['offset_mapping'][0].numpy()
    char_mask = np.zeros(len(raw_path), dtype=bool)
    
    for i, prob in enumerate(probs):
        if prob > THRESHOLD:
            start, end = offset_mapping[i]
            if start == end: continue
            if end <= len(raw_path):
                char_mask[start:end] = True
                
    final_res = ""
    true_indices = np.where(char_mask)[0]
    if len(true_indices) > 0:
        groups = []
        curr_grp = [true_indices[0]]
        for i in range(1, len(true_indices)):
            prev, curr = true_indices[i-1], true_indices[i]
            gap_str = raw_path[prev+1:curr]
            if (curr - prev) < 5 and not any(c.isalnum() for c in gap_str):
                curr_grp.append(curr)
            else:
                groups.append(curr_grp)
                curr_grp = [curr]
        groups.append(curr_grp)
        
        best_grp = max(groups, key=len)
        raw_extract = raw_path[best_grp[0]:best_grp[-1] + 1]
        final_res = TextUtils.cleanup_result(raw_extract)

    print(f"{raw_path}#{final_res}" if final_res else f"{raw_path}#")

def load_model_for_inference():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print("请先运行训练 (python main.py train)")
        return None, None
    
    # 关键：预测时也直接加载本地的基础模型，不联网
    bert_path = get_bert_path()
    
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = BertExtractor(bert_path)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'train':
            run_train()
        else:
            model, tok = load_model_for_inference()
            if model:
                if os.path.isfile(cmd):
                    with open(cmd, 'r', encoding='utf-8') as f:
                        for l in f: predict_single(l, model, tok)
                else:
                    predict_single(cmd, model, tok)
    else:
        run_train()