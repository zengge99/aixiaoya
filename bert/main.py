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
# 屏蔽 Transformers 的啰嗦警告
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 核心配置
NUM_THREADS = 4       # CPU 线程数
BATCH_SIZE = 32       # 批次大小 (CPU 建议 32-64)
LR = 5e-5             # 学习率
EPOCHS = 10           # 训练轮数
MAX_LEN = 128         # 序列截断长度 (128足够覆盖绝大多数路径)
MODEL_NAME = "prajjwal1/bert-tiny"  # TinyBERT: 2层, 128维, 速度极快
MODEL_PATH = "movie_model_bert.pth"
DATA_FILE_PATTERN = "train_data*.txt"
SEED = 42

# 预测配置
THRESHOLD = 0.5       # 判定阈值

torch.set_num_threads(NUM_THREADS)

# --- 工具类 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TextUtils:
    @staticmethod
    def preprocess_for_bert(text):
        """
        关键逻辑：
        1. 必须保持字符串长度不变 (1对1替换)，以便 offset_mapping 映射回原字符串。
        2. 将 . 和 _ 替换为空格 (它们通常是单词分隔符)。
        3. 保留 [] () - ' : 等有助于判断结构的语义符号。
        4. 其他乱七八糟的符号替换为空格。
        """
        if not text: return ""
        
        # 1. 核心分隔符处理：点和下划线 -> 空格
        # 这让 "Iron.Man" 变成 "Iron Man"，BERT 更容易理解
        text = re.sub(r'[._]', ' ', text)
        
        # 2. 定义白名单 (保留字符)
        # a-z, 0-9, 中文
        chars = r'a-zA-Z0-9\u4e00-\u9fa5'
        # 结构性标点 (保留它们，因为 BERT 知道 [ ] 里面的通常不是正文)
        puncts = r'\[\]\(\)\{\}\-\'\"\:!&' 
        # 中文标点
        cn_puncts = r'【】（）《》：'
        
        # 3. 将所有“非白名单”字符替换为空格
        pattern = f'[^{chars}{puncts}{cn_puncts}\s]'
        text = re.sub(pattern, ' ', text)
        
        return text

    @staticmethod
    def cleanup_result(text):
        """结果清洗：去头去尾，规范化空格"""
        if not text: return ""
        # 去除首尾的标点和空格
        text = text.strip(" .-_[]()/\\")
        # 把中间的连续空格合并
        text = re.sub(r'\s+', ' ', text)
        return text

# --- 模型定义 ---
class BertExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # TinyBERT 的 hidden_size 是 128
        self.hidden_size = self.bert.config.hidden_size 
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        # BERT 输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取最后一层的序列输出: [batch, seq_len, hidden]
        sequence_output = outputs.last_hidden_state
        
        # 映射到概率: [batch, seq_len, 1] -> [batch, seq_len]
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
            
            # 简单验证：确保 target 在 input 里能找到
            # 这里做一个宽容匹配，因为路径里的目标可能带点，标签里可能是空格
            escaped = re.escape(target_name).replace(r'\ ', r'.*')
            if re.search(escaped, input_path, re.IGNORECASE):
                self.samples.append((input_path, target_name))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_name = self.samples[idx]
        
        # 1. 确定目标在原始字符串中的 Start/End 索引
        # 我们用正则找最后一次出现的匹配
        escaped_target = re.escape(target_name)
        # 允许目标名中间的空格在路径里是任意字符 (比如 Iron Man -> Iron.Man)
        pattern = escaped_target.replace(r'\ ', r'.+') 
        
        matches = list(re.finditer(pattern, input_path, re.IGNORECASE))
        if matches:
            match = matches[-1]
            start_char, end_char = match.start(), match.end()
        else:
            start_char, end_char = -1, -1

        # 2. 预处理文本 (保留特殊符号，去除干扰符)
        text_for_bert = TextUtils.preprocess_for_bert(input_path)

        # 3. Tokenize
        # return_offsets_mapping=True 是核心，它告诉我们 token 对应原文本哪里
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

        # 4. 生成 Label
        labels = torch.zeros(self.max_len, dtype=torch.float)
        
        if start_char != -1:
            for i, (start, end) in enumerate(offset_mapping):
                if attention_mask[i] == 0: continue 
                if start == end: continue # 跳过 [CLS] [SEP]
                
                # 计算重叠
                overlap_start = max(start, start_char)
                overlap_end = min(end, end_char)
                overlap = max(0, overlap_end - overlap_start)
                token_len = end - start
                
                # 如果 token 超过 40% 的部分属于电影名，或者是被电影名完全包含
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
        # 不计算 padding 部分的 loss
        loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # 增加正样本(电影名)的权重
        weights = targets * self.pos_weight + (1 - targets)
        loss = loss * weights * mask
        return loss.sum() / (mask.sum() + 1e-6)

# --- 训练流程 ---
def run_train():
    set_seed(SEED)
    print(f"正在下载/加载模型: {MODEL_NAME} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"模型下载失败，请检查网络。错误: {e}")
        return

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
    
    model = BertExtractor(MODEL_NAME)
    
    # 如果有旧模型，加载继续练
    if os.path.exists(MODEL_PATH):
        print("加载已有模型权重...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = WeightedBCELoss(pos_weight=6.0)
    
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
            
            preds = model(input_ids, mask)
            loss = criterion(preds, labels, mask)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # 验证
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
            torch.save(model.state_dict(), MODEL_PATH)
            print(" >> 模型已保存 (Best)")

# --- 预测流程 ---
def predict_single(raw_path, model, tokenizer):
    if not raw_path.strip() or raw_path.startswith('#'): return
    raw_path = raw_path.strip()

    # 1. 预处理
    text_for_bert = TextUtils.preprocess_for_bert(raw_path)
    
    # 2. 推理
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
    
    # 3. 映射回字符 mask
    char_mask = np.zeros(len(raw_path), dtype=bool)
    for i, prob in enumerate(probs):
        if prob > THRESHOLD:
            start, end = offset_mapping[i]
            if start == end: continue
            if end <= len(raw_path):
                char_mask[start:end] = True
                
    # 4. 提取与合并 (Gap Filling)
    # 如果提取出的字符中间只隔了 1-2 个字符，且那些字符是标点/空格，则连起来
    # 例如: "Iron" [True] "." [False] "Man" [True] -> "Iron.Man"
    
    final_res = ""
    true_indices = np.where(char_mask)[0]
    
    if len(true_indices) > 0:
        groups = []
        curr_grp = [true_indices[0]]
        
        for i in range(1, len(true_indices)):
            prev = true_indices[i-1]
            curr = true_indices[i]
            
            # Gap Filling 逻辑：
            # 如果中间断开小于4个字符，并且断开的部分没有字母数字（只是符号）
            # 或者断开部分包含 & : - 等强连接符
            gap_len = curr - prev
            gap_str = raw_path[prev+1:curr]
            
            is_gap_safe = not any(c.isalnum() for c in gap_str) # 只有标点
            
            if gap_len < 5 and is_gap_safe:
                curr_grp.append(curr)
            else:
                groups.append(curr_grp)
                curr_grp = [curr]
        groups.append(curr_grp)
        
        # 取最长的一组 (通常电影名最长)
        best_grp = max(groups, key=len)
        
        # 从该组的第一个字符到最后一个字符，从原字符串截取
        start_idx = best_grp[0]
        end_idx = best_grp[-1] + 1
        raw_extract = raw_path[start_idx:end_idx]
        
        final_res = TextUtils.cleanup_result(raw_extract)

    # 输出
    if final_res:
        print(f"{raw_path}#{final_res}")
    else:
        print(f"{raw_path}#")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("请先运行训练 (python main.py train)")
        return None, None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BertExtractor(MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'train':
            run_train()
        else:
            model, tok = load_model()
            if model:
                if os.path.isfile(cmd):
                    with open(cmd, 'r', encoding='utf-8') as f:
                        for l in f: predict_single(l, model, tok)
                else:
                    predict_single(cmd, model, tok)
    else:
        run_train()