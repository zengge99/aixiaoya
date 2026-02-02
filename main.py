import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import sys
import os
import re
import random
import numpy as np
import glob
import string

# --- 全局核心配置 ---
NUM_THREADS = 4
BATCH_SIZE = 128
LR = 1e-6            # 学习率
EPOCHS = 50          # 训练轮数
MAX_LEN = 300        # 最大序列长度
EMBED_DIM = 64      # 向量维度
HIDDEN_DIM = 96     # 隐藏层维度

MODEL_PATH = "movie_model.pth"
VOCAB_PATH = "vocab.pkl"
# 数据文件匹配模式 (匹配 train_data.txt, train_data_2.txt 等)
DATA_FILE_PATTERN = "train_data*.txt" 
SEED = 42            # 固定随机种子

# --- 预测/调试配置 ---
DEBUG_MODE = False    # 开启调试详情
THRESHOLD = 0.35     # 判定阈值
SMOOTH_VAL = 0.1     # 平滑救回阈值

# 设置线程数
torch.set_num_threads(NUM_THREADS)

# --- 辅助工具类 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 移植的 JS 逻辑工具类 ---
class TextUtils:
    CN_NUMS = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]

    @staticmethod
    def number2text(text):
        if not text: return text
        text = text.lstrip('0')
        if not text: return "零" 

        try:
            num = int(text)
        except ValueError:
            return text

        if num <= 10:
            return TextUtils.CN_NUMS[num]
        elif num < 20:
            return "十" + TextUtils.CN_NUMS[num % 10]
        elif num % 10 == 0:
            return TextUtils.CN_NUMS[num // 10] + "十"
        else:
            return TextUtils.CN_NUMS[num // 10] + "十" + TextUtils.CN_NUMS[num % 10]

    @staticmethod
    def fix_name_internal(path, ai_result):
        # 判断是否为全英文（包含字母、空格、数字、常见标点，无中文字符）
        if ai_result and all(ord(c) < 128 for c in ai_result):
            return ai_result

        replace_patterns = [
            r'Season\s*(\d{1,2})',              
            r'SE(\d{1,2})',                     
            r'(?<![a-zA-Z])S(\d{1,2})(?![a-zA-Z])', 
            r'第(\d{1,2})季'                    
        ]

        processed_result = ai_result
        replaced_flag = False 

        def replace_func(match):
            nonlocal replaced_flag
            replaced_flag = True
            num = match.group(1)
            cn_num = TextUtils.number2text(num)
            return f" 第{cn_num}季 " 

        for pattern in replace_patterns:
            if re.search(pattern, processed_result, re.IGNORECASE):
                processed_result = re.sub(pattern, replace_func, processed_result, flags=re.IGNORECASE)
        
        processed_result = re.sub(r'\s+', ' ', processed_result).strip()

        if replaced_flag:
            return processed_result

        path_search_patterns = [
            r'Season\s*(\d{1,2})',
            r'SE(\d{1,2})',
            r'第(\d{1,2})季',
            r'(?<![A-Za-z])S(\d{1,2})'
        ]

        for pattern in path_search_patterns:
            match = re.search(pattern, path, re.IGNORECASE)
            if match:
                num = match.group(1)
                cn_num = TextUtils.number2text(num)
                suffix = f"第{cn_num}季"
                if suffix not in processed_result:
                    return f"{processed_result} {suffix}"
                break 
        
        return processed_result

    @staticmethod
    def fix_name(path, ai_result):
        return TextUtils.fix_name_internal(path, ai_result).replace("第一季", "", 1).strip()

class Extractor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, 
                            batch_first=True, num_layers=2, dropout=0.5)
        
        self.attention_linear = nn.Linear(hidden_dim * 2, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        emb = self.embedding(x)
        
        cnn_in = emb.permute(0, 2, 1)
        cnn_out = self.conv1(cnn_in)
        cnn_out = self.relu(cnn_out).permute(0, 2, 1)
        
        rnn_in = self.norm1(emb + cnn_out)
        
        h0 = torch.zeros(2 * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(2 * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(rnn_in, (h0, c0)) 
        
        attn_scores = torch.tanh(self.attention_linear(lstm_out))
        attn_weights = F.softmax(attn_scores, dim=1)
        
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        seq_len = lstm_out.size(1)
        context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)
        
        combined = torch.cat([lstm_out, context_expanded], dim=2)
        
        return self.fc(combined).squeeze(-1)

# --- 标准加权交叉熵 Loss (比 Focal Loss 更稳健) ---
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=4.0, reduction='mean'):
        """
        Args:
            pos_weight (float): 正样本(电影名)的权重倍数。
                                默认为 4.0，意味着一个电影名字符的重要性是一个背景字符的 4 倍。
            reduction (str): 'mean' 或 'sum'
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. 计算标准的二分类交叉熵 (不求平均，保留每个点的 loss)
        # inputs: 模型输出的概率 (0~1)
        # targets: 真实标签 (0 或 1)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # 2. 生成权重向量
        # 如果 target 是 1 (正样本)，权重 = self.pos_weight
        # 如果 target 是 0 (负样本)，权重 = 1.0
        weights = targets * self.pos_weight + (1 - targets)

        # 3. 加权
        weighted_loss = bce_loss * weights

        # 4. 归约返回
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha   # 增大正样本(电影名字符)的权重
        self.gamma = gamma   # 聚焦难分类样本
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        return loss

# --- 数据集定义 ---
class MovieDataset(Dataset):
    PREFIX_LIST = [
        'Download', 'Downloads', 'Movies', 'TV_Shows', 'Media', 'Video', 
        'Temp', 'Backup', 'Data', 'New_Folder', 'Private',
        '下载', '迅雷下载', '百度网盘', '我的下载', '传输文件', '接收文件',
        '电影', '电视剧', '视频', '影视', '剧集', '动漫', '动画', 
        '综艺', '纪录片', '合集', '国产剧', '美剧', '日剧', '韩剧',
        '新建文件夹', '备份', '临时', '资料', '桌面', '我的文档', 
        '收藏', '待看', '已看', '整理', '回收站'
    ]
    
    SUFFIX_LIST = [
        '1080p', '720p', '2160p', '4K', 'x264', 'x265', 'HEVC', 
        'HDR', 'BluRay', 'BDrip', 'WebDL', 'HDTV', 'AAC', 'DTS', 'Atmos',
        '高清', '蓝光', '字幕', '福利', '完结', '未删减', '加长版',
        '中字', '双语', '国语', '特效字幕', '含花絮', '修正版',
        'Backup', 'Copy', 'Temp', '副本'
    ]
    def __init__(self, lines, char_to_idx, max_len=MAX_LEN, training=True):
        self.samples = []
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        self.training = training  # 控制是否开启随机增强
        
        skipped_count = 0
        
        # 1. 在 Init 中仅做有效性筛选，保存原始文本
        for line in lines:
            line = line.strip()
            if '#' not in line: continue
            input_path, target_name = line.rsplit('#', 1)
            target_name = target_name.strip()
            
            # 预检查：确保原始数据是能匹配上的
            escaped_target = re.escape(target_name)
            pattern = escaped_target.replace(r'\ ', r'[._\s]+')
            if re.search(pattern, input_path, re.IGNORECASE):
                self.samples.append((input_path, target_name))
            else:
                skipped_count += 1

        if skipped_count > 0:
            print(f"Dataset Info: 跳过了 {skipped_count} 条无法匹配标签的数据。")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        # 2. 获取原始数据
        input_path, target_name = self.samples[idx]
        
        # 3. 随机路径增强 (仅在训练模式下)
        if random.random() < 0.3:
            # 50% 概率加前缀（模拟目录），50% 概率加后缀（模拟文件属性）
            if random.random() < 0.5:
                # --- 情况 1：加前缀 (模拟父级目录) ---
                noise = random.choice(self.PREFIX_LIST)
                input_path = f"{noise}/{input_path}"
            else:
                # --- 情况 2：加后缀 (模拟文件属性/标签) ---
                noise = random.choice(self.SUFFIX_LIST)
                input_path = f"{input_path}/{noise}"

            # if random.random() < 0.3:
            #     noise1 = random.choice(['01', '02', 'S01E01', 'S01E02', '1', '2', '3', 'S02E01'])
            #     noise2 = random.choice(['.mp4', '.mkv', '.avi', '.rmvb', '.wmv', '.mov', '.flv', '.iso'])
            #     noise = f"{noise1}{noise2}"
            #     input_path = f"{input_path}/{noise}"

            # === Part B: 分隔符扰动 ===
            if random.random() < 0.3:
                input_path = input_path.replace('.', ' ')
            elif random.random() < 0.3:
                input_path = input_path.replace('_', ' ')
            elif random.random() < 0.2:
                input_path = input_path.replace(' ', '.')

        # 4. 实时计算索引 (匹配最后一次出现的目标名，优先文件名)
        escaped_target = re.escape(target_name)
        pattern = escaped_target.replace(r'\ ', r'[._\s]+')

        # 找所有匹配项，取最后一个
        matches = list(re.finditer(pattern, input_path, re.IGNORECASE))
        match = matches[-1] if matches else None

        # 兜底：如果匹配失败（极少见），回退到原始路径重新匹配
        if not match:
            input_path, _ = self.samples[idx]
            matches = list(re.finditer(pattern, input_path, re.IGNORECASE))
            match = matches[-1] if matches else None

        start_idx = match.start()
        end_idx = match.end()
        
        # 5. 转 Tensor 和 Padding
        # 截断输入，防止增强后长度溢出
        input_ids = [self.char_to_idx.get(c.lower(), 1) for c in input_path[:self.max_len]]
        labels = [0.0] * len(input_ids)
        
        limit = min(end_idx, self.max_len)
        for i in range(start_idx, limit):
            labels[i] = 1.0
        
        pad_len = self.max_len - len(input_ids)
        
        # 确保 pad_len 不为负数
        pad_len = max(0, pad_len)
        
        return (
            torch.tensor(input_ids + [0] * pad_len), 
            torch.tensor(labels + [0.0] * pad_len)
        )

# --- 辅助函数：验证集计算 ---
def validate_one_epoch(model, loader, criterion):
    model.eval()
    v_loss = 0
    with torch.no_grad():
        for vx, vy in loader:
            pred = model(vx)
            loss = criterion(pred, vy)
            v_loss += loss.item()
    return v_loss / len(loader) if len(loader) > 0 else 0

# --- 训练逻辑 ---
def run_train(incremental=False):
    # set_seed(SEED)
    mode_str = "【增量训练模式】" if incremental else "【全量训练模式】"
    print(f"{mode_str}")

    # 1. 搜索所有匹配的文件
    data_files = glob.glob(DATA_FILE_PATTERN)
    # 排序以保证每次运行读取顺序一致，确保 index 0 总是同一个文件
    data_files.sort()
    
    if not data_files:
        print(f"❌ 未找到匹配 {DATA_FILE_PATTERN} 的数据文件。"); return
    
    print(f"发现 {len(data_files)} 个数据文件: {data_files}")

    all_train_lines = []
    all_val_lines = []
    
    # 使用独立的 Random 实例进行 shuffle，不影响全局状态
    rng = random.Random(SEED)
    
    # 2. 遍历每个文件
    for i, f_path in enumerate(data_files):
        with open(f_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if '#' in l.strip()]
        
        # 步骤A: 确定性打乱
        rng.shuffle(lines)
        total_raw = len(lines)
        if total_raw == 0: continue
        
        # 步骤B: 先进行 训练/验证 切分
        # 无论是否增量，永远固定前90%为训练池，后10%为验证池。
        # 这样可以保证验证集永远纯净，不会因为增量裁切导致训练数据越界。
        split_idx = int(total_raw * 0.9)
        if split_idx == 0 and total_raw > 0: split_idx = total_raw # 极少数据保护
        
        file_train_lines = lines[:split_idx]
        file_val_lines = lines[split_idx:]
        
        # 步骤C: 处理增量逻辑（仅在切分后的各自池子内进行保留/丢弃）
        if incremental and i == 0:
            # 旧文件：仅保留 2% 的训练数据，以及 2% 的验证数据 (保持分布一致，且节省验证时间)
            keep_train_count = int(len(file_train_lines) * 0.02)
            keep_val_count = int(len(file_val_lines) * 0.02)
            
            # 最小保留保护
            if keep_train_count == 0 and len(file_train_lines) > 0: keep_train_count = 1
            if keep_val_count == 0 and len(file_val_lines) > 0: keep_val_count = 1
            
            final_train = file_train_lines[:keep_train_count]
            final_val = file_val_lines[:keep_val_count]
            
            print(f"  └─ [Old Data] {os.path.basename(f_path)}: 采样保留 训练{len(final_train)}条 / 验证{len(final_val)}条")
        else:
            # 新文件或全量模式：保留切分后的所有数据
            final_train = file_train_lines
            final_val = file_val_lines
            print(f"  └─ [New Data] {os.path.basename(f_path)}: 全量读取 训练{len(final_train)}条 / 验证{len(final_val)}条")

        all_train_lines.extend(final_train)
        all_val_lines.extend(final_val)

    print(f"\n数据集准备完毕: 训练集 {len(all_train_lines)} 条 | 验证集 {len(all_val_lines)} 条")

    # 4. 构建或加载词表
    all_lines_for_vocab = all_train_lines + all_val_lines
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'rb') as f: char_to_idx = pickle.load(f)
        print("已加载现有词表。")
    else:
        # 强行注入基础 ASCII 字符，防止英文词表缺失
        raw_paths = [l.split('#')[0] for l in all_train_lines + all_val_lines]
        all_chars = set("".join(raw_paths).lower())
        
        ascii_chars = set(string.ascii_lowercase + string.digits + string.punctuation + " ")
        all_chars.update(ascii_chars)
        
        char_to_idx = {c: i+2 for i, c in enumerate(sorted(list(all_chars)))}
        char_to_idx['<PAD>'], char_to_idx['<UNK>'] = 0, 1
        with open(VOCAB_PATH, 'wb') as f: pickle.dump(char_to_idx, f)
        print(f"已创建新词表，包含 {len(char_to_idx)} 个字符。")

    # 5. 创建 Dataset 和 DataLoader
    train_ds = MovieDataset(all_train_lines, char_to_idx, training=True)
    val_ds = MovieDataset(all_val_lines, char_to_idx, training=False)


    if len(train_ds) < 1:
        print("有效样本数量不足，无法进行训练。"); return
    # g = torch.Generator()
    # g.manual_seed(SEED)
    #train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=min(4, NUM_THREADS))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=min(4, NUM_THREADS))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=min(4, NUM_THREADS))

    # 初始化新模型
    model = Extractor(len(char_to_idx), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    
    # 使用 Focal Loss
    #criterion = FocalLoss(alpha=0.75, gamma=2)
    criterion = WeightedBCELoss(pos_weight=4.0, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5) # 增加 weight_decay 防止过拟合
    
    best_val_loss = float('inf')

    # 加载模型逻辑
    if os.path.exists(MODEL_PATH):
        print(f"检测到现有模型，加载权重以 LR={LR} 继续微调...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        
        if len(val_ds) > 0:
            print("正在计算当前模型的初始验证集 Loss (基准线)...")
            initial_val_loss = validate_one_epoch(model, val_loader, criterion)
            best_val_loss = initial_val_loss 
            print(f"当前模型基准 Loss: {best_val_loss:.4f}")
    else:
        print("未检测到模型，将从头开始训练。")
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}")
            for x, y in pbar:
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                # 梯度裁剪，防止爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            if len(val_ds) > 0:
                avg_val_loss = validate_one_epoch(model, val_loader, criterion)
                
                if avg_val_loss < best_val_loss:
                    print(f" ✨ 验证集 Loss 优化 ({best_val_loss:.4f} -> {avg_val_loss:.4f})，模型已更新。")
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), MODEL_PATH)
                else:
                    print(f" ⏳ 验证集 Loss: {avg_val_loss:.4f} (未提升，最佳: {best_val_loss:.4f})")
            else:
                torch.save(model.state_dict(), MODEL_PATH)
                print(" ⚠️ 无验证集，模型已保存。")
                
    except KeyboardInterrupt: print("\n🛑 用户手动停止训练。")

# --- 一次性初始化模型 ---
def init_model_and_vocab():
    """初始化模型和词表，返回 (model, char_to_idx)，批量预测时仅调用1次"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print("错误: 找不到模型或词表文件。请先运行训练。")
        return None, None
    
    # 加载词表（仅1次）
    with open(VOCAB_PATH, 'rb') as f:
        char_to_idx = pickle.load(f)
    
    # 初始化模型 + 加载权重（仅1次）
    model = Extractor(len(char_to_idx), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()  # 预测模式，禁用Dropout/BatchNorm
    
    return model, char_to_idx

def predict_single_path(path, model, char_to_idx):
    """单条路径预测，复用已初始化的模型"""
    if '#' in path:
        print(path)
        return
    
    # 输入预处理（仅处理当前路径，无重复加载）
    input_ids = [char_to_idx.get(c.lower(), 1) for c in path[:MAX_LEN]]
    padded = input_ids + [0] * (MAX_LEN - len(input_ids))
    
    with torch.no_grad():
        probs = model(torch.tensor([padded]))[0][:len(path)].numpy()

    # --- 后处理策略 ---
    selected_mask = [False] * len(probs)
    
    # 1. 阈值筛选
    for i, p in enumerate(probs):
        if p > THRESHOLD: selected_mask[i] = True
            
    # 2. 空洞填补 (Gap Filling) - 修复 "Iron.Man" 中间断开的问题
    gap_limit = 2 
    for i in range(len(probs)):
        if selected_mask[i]:
            # 寻找下一个被选中的点
            for j in range(i + 1, min(i + gap_limit + 2, len(probs))):
                if selected_mask[j]:
                    # 将中间所有非路径分隔符的字符都连起来
                    for k in range(i + 1, j):
                        if path[k] not in ['/', '\\']:
                            selected_mask[k] = True
                    break

    res_list = []

    if DEBUG_MODE:
        print(f"\n{'='*65}")
        print(f"{'索引':<4} | {'字符':<4} | {'分值':<15} | 状态")
        print("-" * 65)
        for i, p in enumerate(probs):
            status = "✅ [选中]" if p > THRESHOLD else "   [排除]"
            print(f"{i:<4} | {path[i]:<4} | {p:.10f} | {status}")
        print(f"{'='*65}\n")

    
    for i, is_sel in enumerate(selected_mask):
        if is_sel:
            res_list.append(path[i])

    raw_result = "".join(res_list)
    clean_result = raw_result.replace('.', ' ').replace('_', ' ')
    clean_result = re.sub(r'\s+', ' ', clean_result)
    clean_result = clean_result.strip("/()# “”.-")

    # 1. 验证连续性
    if clean_result:
        escaped_clean = re.escape(clean_result)
        verify_pattern = escaped_clean.replace(r'\ ', r'[._\s\-\(\)\[\]]*')
        if not re.search(verify_pattern, path, re.IGNORECASE):
            if DEBUG_MODE:
                print(f"[验证失败] '{clean_result}' 无法在原路径中连续匹配，判定为无效提取。")
            clean_result = ""

    # 2. 混合模式修复
    if clean_result:
        clean_result = TextUtils.fix_name(path, clean_result) 

    if DEBUG_MODE: 
        print(f"提取原文: {raw_result}")
        print(f"最终结果: {clean_result}")
    else: 
        if clean_result:
            print(f"{path}#{clean_result}")
        else:
            print(f"{path}#")

def run_batch_predict(file_path):
    # 1. 一次性初始化模型和词表
    model, char_to_idx = init_model_and_vocab()
    if model is None or char_to_idx is None:
        return
    
    # 2. 读取文件中的所有路径
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 3. 循环预测所有路径
    total_lines = len(lines)
    for idx, line in enumerate(lines):
        predict_single_path(line, model, char_to_idx)

# --- 入口控制 ---
if __name__ == "__main__":
    if os.path.exists("dbg"):
        DEBUG_MODE = True
        print(f"检测到 [dbg] 文件，已强制开启调试模式")

    if len(sys.argv) > 1:
        input_arg = sys.argv[1]

        if input_arg == '--inc':
            # 模式 1: 增量训练
            run_train(incremental=True)
        
        elif os.path.exists(input_arg) and os.path.isfile(input_arg):
            # 模式 2: 批量预测
            run_batch_predict(input_arg)
        
        else:
            # 模式 3: 单条字符串预测
            model, char_to_idx = init_model_and_vocab()
            if model and char_to_idx:
                predict_single_path(input_arg, model, char_to_idx)
    else:
        # 模式 4: 默认全量训练
        run_train(incremental=False)
