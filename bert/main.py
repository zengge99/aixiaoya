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
# 引入特定类
from transformers import BertTokenizer, BertForTokenClassification, logging
import warnings

# --- 1. 全局配置区 ---
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 硬件与训练参数
NUM_THREADS = 4        
BATCH_SIZE = 32        
LR = 5e-5              
EPOCHS = 10            
MAX_LEN = 128          

# === 模型选择 (用户指定) ===
# 华为官方 TinyBERT (通用版 4层)
BERT_HF_NAME = 'huawei-noah/TinyBERT_General_4L_312D'
BERT_LOCAL_FOLDER = "bert_base_huawei_general"

# 权重文件
MODEL_WEIGHTS_PATH = "movie_model_bio.pth"
DATA_FILE_PATTERN = "train_data*.txt"

SEED = 42
# 0=Outside, 1=Begin, 2=Inside
NUM_LABELS = 3 

torch.set_num_threads(NUM_THREADS)

# --- 2. 辅助工具类 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TextUtils:
    @staticmethod
    def preprocess_for_bert(text):
        if not text: return ""
        # 1. 分隔符处理
        text = re.sub(r'[._]', ' ', text)
        # 2. 白名单保留
        chars = r'a-zA-Z0-9\u4e00-\u9fa5'
        puncts = r'\[\]\(\)\{\}\-\'\"\:!&' 
        cn_puncts = r'【】（）《》：'
        # 3. 替换非法字符
        pattern = f'[^{chars}{puncts}{cn_puncts}\\s]'
        text = re.sub(pattern, ' ', text)
        return text

    @staticmethod
    def cleanup_result(text):
        if not text: return ""
        text = text.strip(" .-_[]()/\\")
        text = re.sub(r'\s+', ' ', text)
        return text

# --- 3. 自动下载逻辑 ---
def get_bert_path():
    if os.path.exists(BERT_LOCAL_FOLDER) and os.listdir(BERT_LOCAL_FOLDER):
        return BERT_LOCAL_FOLDER
    else:
        print(f"⚠️  本地未检测到模型，正在从 {BERT_HF_NAME} 下载...")
        try:
            tokenizer = BertTokenizer.from_pretrained(BERT_HF_NAME)
            # 注意：这里下载基础模型时，先不加 num_labels，仅保存架构
            model = BertForTokenClassification.from_pretrained(BERT_HF_NAME, num_labels=NUM_LABELS)
            
            os.makedirs(BERT_LOCAL_FOLDER, exist_ok=True)
            tokenizer.save_pretrained(BERT_LOCAL_FOLDER)
            model.save_pretrained(BERT_LOCAL_FOLDER)
            
            print(f"✅ 模型已下载并保存。")
            return BERT_LOCAL_FOLDER
        except Exception as e:
            print(f"\n❌ 下载失败: {e}")
            print("💡 提示: 请尝试设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
            sys.exit(1)

# --- 4. 模型封装 (Wrapper) ---
class NERModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # 直接加载 BertForTokenClassification
        # num_labels=3 (O, B-Movie, I-Movie)
        self.bert = BertForTokenClassification.from_pretrained(model_path, num_labels=NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        # return_dict=True 返回对象，我们只需要 logits
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return output.logits  # Shape: [batch, seq_len, 3]

# --- 5. 数据集 (BIO 标注) ---
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
        
        # 1. 找位置
        escaped_target = re.escape(target_name)
        pattern = escaped_target.replace(r'\ ', r'.+') 
        matches = list(re.finditer(pattern, input_path, re.IGNORECASE))
        
        start_char, end_char = -1, -1
        if matches:
            match = matches[-1]
            start_char, end_char = match.start(), match.end()

        # 2. Tokenize
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

        # 3. 生成 Labels (Long Tensor)
        # 0: O (Outside)
        # 1: B (Begin)
        # 2: I (Inside)
        labels = torch.zeros(self.max_len, dtype=torch.long)
        
        if start_char != -1:
            found_start = False # 标记是否找到了第一个token
            
            for i, (start, end) in enumerate(offset_mapping):
                if attention_mask[i] == 0 or start == end: 
                    labels[i] = -100 # PyTorch CrossEntropyLoss 默认忽略 -100
                    continue
                
                # 计算重叠
                overlap_start = max(start, start_char)
                overlap_end = min(end, end_char)
                overlap = max(0, overlap_end - overlap_start)
                token_len = end - start
                
                # 判定是否属于电影名
                if token_len > 0 and (overlap / token_len > 0.4):
                    if not found_start:
                        labels[i] = 1 # B-Movie (首个 Token)
                        found_start = True
                    else:
                        labels[i] = 2 # I-Movie (后续 Token)
                else:
                    labels[i] = 0 # O

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# --- 6. 训练流程 ---
def run_train():
    set_seed(SEED)
    bert_path = get_bert_path()
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    # 读取数据
    data_files = glob.glob(DATA_FILE_PATTERN)
    if not data_files:
        print(f"❌ 未找到训练数据 {DATA_FILE_PATTERN}"); return
    
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
    
    # 加载模型
    model = NERModel(bert_path)
    if os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"🔄 加载已有权重: {MODEL_WEIGHTS_PATH}")
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # Loss: CrossEntropyLoss
    # class_weights: [O的权重, B的权重, I的权重]
    # O非常多，权重设小点；B最少，权重设大点；I适中
    class_weights = torch.tensor([1.0, 10.0, 8.0]) 
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            mask = batch['attention_mask']
            labels = batch['labels']
            
            # Forward: [batch, seq, 3]
            logits = model(input_ids, mask)
            
            # Flatten 之后计算 Loss
            # logits: [batch*seq, 3], labels: [batch*seq]
            loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['input_ids'], batch['attention_mask'])
                loss = criterion(logits.view(-1, NUM_LABELS), batch['labels'].view(-1))
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"   └─ Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            print(f"   ✨ 模型已保存")

# --- 7. 预测流程 (适配 BIO) ---
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
        # logits: [1, seq_len, 3]
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        # 取最大概率的索引: [1, seq_len] -> [seq_len]
        preds = torch.argmax(logits, dim=2)[0].numpy()
        
    offset_mapping = inputs['offset_mapping'][0].numpy()
    
    # 还原回字符级 Mask
    char_mask = np.zeros(len(raw_path), dtype=bool)
    
    # Debug 信息
    debug_info = []

    for i, pred_class in enumerate(preds):
        start, end = offset_mapping[i]
        if start == end: continue
        if end > len(raw_path): continue
        
        # 只要是 B(1) 或 I(2)，都认为是实体的一部分
        if pred_class in [1, 2]:
            char_mask[start:end] = True
            
        if os.path.exists("dbg"):
            token_str = raw_path[start:end]
            label_str = "O" if pred_class == 0 else ("B" if pred_class == 1 else "I")
            debug_info.append(f"{token_str:<5} | {label_str}")

    # Debug 打印
    if os.path.exists("dbg"):
        print(f"\nPATH: {raw_path}")
        print("Tokens:", " ".join(debug_info))
    
    # 后处理：提取连续区域
    final_res = ""
    true_indices = np.where(char_mask)[0]
    
    if len(true_indices) > 0:
        groups = []
        curr_grp = [true_indices[0]]
        for i in range(1, len(true_indices)):
            prev, curr = true_indices[i-1], true_indices[i]
            
            # 如果断开，但距离很近且只是标点，则连起来 (Gap Filling)
            gap_str = raw_path[prev+1:curr]
            if (curr - prev) < 5 and not any(c.isalnum() for c in gap_str):
                curr_grp.append(curr)
            else:
                groups.append(curr_grp)
                curr_grp = [curr]
        groups.append(curr_grp)
        
        # 提取最长的一组
        best_grp = max(groups, key=len)
        raw_extract = raw_path[best_grp[0]:best_grp[-1] + 1]
        final_res = TextUtils.cleanup_result(raw_extract)

    if final_res:
        print(f"{raw_path}#{final_res}")
    else:
        print(f"{raw_path}#")

def load_inference_components():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ 未找到权重 {MODEL_WEIGHTS_PATH}，请先训练。")
        return None, None
    
    bert_path = get_bert_path()
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = NERModel(bert_path)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'train':
            run_train()
        else:
            model, tok = load_inference_components()
            if model:
                if os.path.isfile(cmd):
                    with open(cmd, 'r', encoding='utf-8') as f:
                        for l in f: predict_single(l, model, tok)
                else:
                    predict_single(cmd, model, tok)
    else:
        run_train()