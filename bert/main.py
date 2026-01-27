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
# 1. 屏蔽 HuggingFace 的啰嗦警告
logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 2. 核心配置
NUM_THREADS = 4       
BATCH_SIZE = 32       
LR = 5e-5             
EPOCHS = 10           
MAX_LEN = 128         

BERT_HF_NAME = "prajjwal1/bert-tiny" 
BERT_LOCAL_FOLDER = "bert_base_local"
MODEL_WEIGHTS_PATH = "movie_model_bert.pth"
DATA_FILE_PATTERN = "train_data*.txt"
SEED = 42

# 3. 预测配置
THRESHOLD = 0.5       
DEBUG_MODE = False    # 默认关闭，可以通过在目录下创建 'dbg' 文件来开启

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
        # 1. 消除分隔符干扰
        text = re.sub(r'[._]', ' ', text)
        # 2. 保留白名单字符，其他变空格 (保持长度不变)
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

# --- 自动下载逻辑 ---
def get_bert_path():
    if os.path.exists(BERT_LOCAL_FOLDER) and os.listdir(BERT_LOCAL_FOLDER):
        return BERT_LOCAL_FOLDER
    else:
        print(f"正在从 {BERT_HF_NAME} 下载基础模型...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(BERT_HF_NAME)
            model = AutoModel.from_pretrained(BERT_HF_NAME)
            os.makedirs(BERT_LOCAL_FOLDER, exist_ok=True)
            tokenizer.save_pretrained(BERT_LOCAL_FOLDER)
            model.save_pretrained(BERT_LOCAL_FOLDER)
            return BERT_LOCAL_FOLDER
        except Exception as e:
            print(f"下载失败: {e}")
            sys.exit(1)

# --- 模型定义 ---
class BertExtractor(nn.Module):
    def __init__(self, model_path):
        super().__init__()
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

# --- Loss ---
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=6.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets, mask):
        loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        weights = targets * self.pos_weight + (1 - targets)
        loss = loss * weights * mask
        return loss.sum() / (mask.sum() + 1e-6)

# --- 训练 ---
def run_train():
    set_seed(SEED)
    bert_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(bert_path)

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
    
    model = BertExtractor(bert_path)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print("加载已有权重...")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
        
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = WeightedBCELoss(pos_weight=6.0)
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
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

# --- 预测 (含 Debug 输出) ---
def predict_single(raw_path, model, tokenizer):
    if not raw_path.strip() or raw_path.startswith('#'): return
    raw_path = raw_path.strip()

    # 1. 预处理 & Tokenize
    text_for_bert = TextUtils.preprocess_for_bert(raw_path)
    inputs = tokenizer(
        text_for_bert,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LEN
    )
    
    # 2. 推理
    with torch.no_grad():
        probs = model(inputs['input_ids'], inputs['attention_mask'])[0].numpy()
        
    offset_mapping = inputs['offset_mapping'][0].numpy()
    
    # 3. 将 Token 级概率映射回 字符级
    char_scores = np.zeros(len(raw_path))          # 存储每个字符的分数 (用于Debug)
    char_mask = np.zeros(len(raw_path), dtype=bool) # 存储每个字符是否被选中
    
    for i, prob in enumerate(probs):
        start, end = offset_mapping[i]
        if start == end: continue # 跳过 [CLS], [SEP]
        if end > len(raw_path): continue
        
        # 将该 Token 的概率赋值给对应的字符区间
        # 注意：这里会覆盖，但因为我们是1对1替换，区间通常不会重叠
        char_scores[start:end] = prob
        
        if prob > THRESHOLD:
            char_mask[start:end] = True
    
    # --- Debug 打印区域 ---
    if DEBUG_MODE:
        print(f"\n{'='*60}")
        print(f"原始路径: {raw_path}")
        print("-" * 60)
        print(f"{'Idx':<4} | {'Char':<4} | {'Score':<8} | {'Status'}")
        print("-" * 60)
        
        for i, char in enumerate(raw_path):
            score = char_scores[i]
            # 简单的视觉标记：分数高显示绿色对勾，低则空
            status = "✅" if score > THRESHOLD else "  "
            
            # 过滤掉分数极低的字符显示，防止刷屏（可选）
            # if score < 0.01: continue 
            
            # 打印
            # 处理换行符等不可见字符的显示
            display_char = char if char.isprintable() else '?'
            print(f"{i:<4} | {display_char:<4} | {score:.4f}   | {status}")
            
        print(f"{'='*60}\n")
    # ---------------------

    # 4. 后处理 (Gap Filling)
    final_res = ""
    true_indices = np.where(char_mask)[0]
    
    if len(true_indices) > 0:
        groups = []
        curr_grp = [true_indices[0]]
        for i in range(1, len(true_indices)):
            prev, curr = true_indices[i-1], true_indices[i]
            gap_str = raw_path[prev+1:curr]
            # 如果断开的部分很短且不是字母数字（是标点），则连起来
            if (curr - prev) < 5 and not any(c.isalnum() for c in gap_str):
                curr_grp.append(curr)
            else:
                groups.append(curr_grp)
                curr_grp = [curr]
        groups.append(curr_grp)
        
        best_grp = max(groups, key=len)
        raw_extract = raw_path[best_grp[0]:best_grp[-1] + 1]
        final_res = TextUtils.cleanup_result(raw_extract)

    if final_res:
        print(f"{raw_path}#{final_res}")
    else:
        print(f"{raw_path}#")

def load_model_for_inference():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print("请先训练: python main.py train")
        return None, None
    bert_path = get_bert_path()
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = BertExtractor(bert_path)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    # 检测当前目录是否有 'dbg' 文件，有则开启 Debug 模式
    if os.path.exists("dbg"):
        DEBUG_MODE = True
        print(">> 检测到 'dbg' 文件，已开启调试详情模式 <<")

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