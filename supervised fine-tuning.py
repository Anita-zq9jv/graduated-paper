import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
    BertConfig,
    logging as transformers_logging
)
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Optional
import time
import psutil
import gc
from torch.cuda.amp import autocast, GradScaler
from safetensors.torch import load_file

# Configure logging
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(module)s:%(lineno)d] %(message)s',
    handlers=[
        logging.FileHandler("./logs/finetune_supervised.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SemanticMatching")
transformers_logging.set_verbosity_error()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

# System info
logger.info(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
logger.info(f"System Memory: {psutil.virtual_memory().total / 1024 ** 3:.2f} GB")


class HierarchicalTextEncoder(nn.Module):
    """Hierarchical encoder for long texts, combining paragraph and full-text representations"""
    def __init__(self, bert_model, cls_token_id, sep_token_id):
        super().__init__()
        self.bert = bert_model
        self.config = bert_model.config
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, total_len = input_ids.shape
        max_window_size = min(510, total_len)
        min_window_size = min(128, total_len)
        window_size = min(max_window_size, max(min_window_size, int(total_len * 0.3)))
        stride = max(64, window_size // 2)
        num_windows = max(1, (total_len - window_size) // stride + 1)

        all_embeddings = []
        all_weights = []
        all_attention_weights = []

        for i in range(num_windows + 1):
            start = i * stride
            end = min(start + window_size, total_len)
            if end - start < 10:
                continue

            window_ids = input_ids[:, start:end]
            window_mask = attention_mask[:, start:end]
            window_ids = torch.cat([
                torch.full((batch_size, 1), self.cls_token_id, device=device, dtype=torch.long),
                window_ids,
                torch.full((batch_size, 1), self.sep_token_id, device=device, dtype=torch.long)
            ], dim=1)
            window_mask = torch.cat([
                torch.ones((batch_size, 1), device=device),
                window_mask,
                torch.ones((batch_size, 1), device=device)
            ], dim=1)

            outputs = self.bert(input_ids=window_ids, attention_mask=window_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            cls_embedding = hidden_states[:, 0, :]
            window_weight = (end - start) / total_len * torch.ones(batch_size, device=device)

            attention_weights = self.attention(hidden_states).squeeze(-1)
            attention_weights = attention_weights.masked_fill(window_mask == 0, -1e4)
            attention_weights = F.softmax(attention_weights, dim=-1)
            window_attention = torch.sum(attention_weights.unsqueeze(-1) * hidden_states, dim=1)

            all_embeddings.append(cls_embedding)
            all_weights.append(window_weight)
            all_attention_weights.append(window_attention)

        if not all_embeddings:
            return torch.zeros((batch_size, self.config.hidden_size * 2), device=device)

        weights = torch.stack(all_weights, dim=1).unsqueeze(-1)
        embeddings = torch.stack(all_embeddings, dim=1)
        cls_embedding = torch.sum(embeddings * weights, dim=1) / torch.sum(weights, dim=1)
        attention_embeddings = torch.stack(all_attention_weights, dim=1)
        attention_embedding = torch.sum(attention_embeddings * weights, dim=1) / torch.sum(weights, dim=1)
        return torch.cat([cls_embedding, attention_embedding], dim=-1)


class MultiScaleSimilarityModel(nn.Module):
    """Enhanced multi-scale semantic similarity model"""
    def __init__(self, model_path, tokenizer, adapter_path=None):
        super().__init__()
        self.config = BertConfig.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path, config=self.config)

        if adapter_path:
            logger.info(f"Loading adapters from {adapter_path}")
            self._load_adapters(adapter_path)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.patent_encoder = HierarchicalTextEncoder(self.bert, tokenizer.cls_token_id, tokenizer.sep_token_id)
        self.career_encoder = HierarchicalTextEncoder(self.bert, tokenizer.cls_token_id, tokenizer.sep_token_id)
        input_size = self.bert.config.hidden_size * 2

        # Enhanced projection with more non-linear layers
        self.projection = nn.Sequential(
            nn.Linear(input_size, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        self.normalize = lambda x: F.normalize(x, p=2, dim=1)

        logger.info("Model initialized")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def _load_adapters(self, adapter_path):
        adapter_file = os.path.join(adapter_path, "model.safetensors")
        if os.path.exists(adapter_file):
            adapter_weights = load_file(adapter_file, device="cpu")
            model_dict = self.bert.state_dict()
            pretrained_dict = {k: v for k, v in adapter_weights.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.bert.load_state_dict(model_dict)
            logger.info(f"Loaded adapters from {adapter_file}")
        else:
            logger.warning(f"Safetensors adapter file not found: {adapter_file}")

    def forward(self, patent_ids, patent_mask, career_ids, career_mask):
        patent_emb = self.patent_encoder(patent_ids, patent_mask)
        career_emb = self.career_encoder(career_ids, career_mask)
        patent_proj = self.projection(patent_emb)
        career_proj = self.projection(career_emb)
        patent_proj = self.normalize(patent_proj)
        career_proj = self.normalize(career_proj)
        return torch.cosine_similarity(patent_proj, career_proj, dim=-1)


class DynamicTripletDataset(Dataset):
    """Dataset with hard negative mining"""
    def __init__(self, data_path, tokenizer, career_data_path, max_length=4096, val_ratio=0.2, mode='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        logger.info(f"Loaded {len(self.data)} samples")

        with open(career_data_path, 'r', encoding='utf-8') as f:
            self.career_data = json.load(f)
        self.career_texts = [item["desc"] for item in self.career_data]
        self.career_codes = [item["code"] for item in self.career_data]

        random.shuffle(self.data)
        split_idx = int(len(self.data) * (1 - val_ratio))
        self.data = self.data[:split_idx] if mode == 'train' else self.data[split_idx:]
        logger.info(f"Prepared {len(self.data)} triplets for {mode}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 安全获取字段，并做兜底（防 None / 非字符串）
        patent = item.get("PatentText", "")
        pdesc = item.get("pdesc", "")
        if pdesc is None:
            pdesc = ""
        if patent is None:
            patent = ""
        # 强制转成字符串（避免传入 list/dict/number 等）
        if not isinstance(patent, str):
            patent = str(patent)
        if not isinstance(pdesc, str):
            pdesc = str(pdesc)

        # Hard negative selection：先尝试符合条件的候选集；如果空则退化到任意非空 career 文本
        candidates = [c for c in self.career_texts if isinstance(c, str) and c != pdesc and len(c) > 50]
        if not candidates:
            candidates = [c for c in self.career_texts if isinstance(c, str) and len(c) > 0]
        ndesc = random.choice(candidates) if candidates else ""

        # 显式使用 text= 参数并捕获 tokenizer 的异常，防止 worker 直接崩溃
        try:
            patent_enc = self.tokenizer(text=patent, truncation=False, padding=False, return_tensors="pt", add_special_tokens=False)
            pdesc_enc = self.tokenizer(text=pdesc, truncation=False, padding=False, return_tensors="pt", add_special_tokens=False)
            ndesc_enc = self.tokenizer(text=ndesc, truncation=False, padding=False, return_tensors="pt", add_special_tokens=False)
        except Exception as e:
            # 记录日志以便后续分析问题样本；同时返回空的 tokenization 作为兜底
            logger.warning(f"Tokenizer failed for idx={idx}, patent_len={len(patent)}, pdesc_len={len(pdesc)}, ndesc_len={len(ndesc)}. Error: {e}")
            patent_enc = self.tokenizer(text="", truncation=False, padding=False, return_tensors="pt", add_special_tokens=False)
            pdesc_enc = self.tokenizer(text="", truncation=False, padding=False, return_tensors="pt", add_special_tokens=False)
            ndesc_enc = self.tokenizer(text="", truncation=False, padding=False, return_tensors="pt", add_special_tokens=False)

        # 将返回的 (1, seq_len) Tensor squeeze -> 1D Tensor
        patent_ids = patent_enc["input_ids"].squeeze(0).long()
        patent_mask = patent_enc.get("attention_mask", torch.ones(patent_ids.size(0), dtype=torch.long)).squeeze(0)
        pdesc_ids = pdesc_enc["input_ids"].squeeze(0).long()
        pdesc_mask = pdesc_enc.get("attention_mask", torch.ones(pdesc_ids.size(0), dtype=torch.long)).squeeze(0)
        ndesc_ids = ndesc_enc["input_ids"].squeeze(0).long()
        ndesc_mask = ndesc_enc.get("attention_mask", torch.ones(ndesc_ids.size(0), dtype=torch.long)).squeeze(0)

        return {
            "patent_ids": patent_ids,
            "patent_mask": patent_mask,
            "pdesc_ids": pdesc_ids,
            "pdesc_mask": pdesc_mask,
            "ndesc_ids": ndesc_ids,
            "ndesc_mask": ndesc_mask,
        }

def dynamic_collate_fn(batch):
    collated = {key: [] for key in batch[0].keys()}
    max_len = {k: max(len(item[k]) for item in batch) for k in ["patent_ids", "pdesc_ids", "ndesc_ids"]}

    for item in batch:
        for k in ["patent_ids", "pdesc_ids", "ndesc_ids"]:
            pad_len = max_len[k] - len(item[k])
            collated[k].append(F.pad(item[k], (0, pad_len), value=0))
            mask_key = k.replace("_ids", "_mask")
            collated[mask_key].append(F.pad(item[mask_key], (0, pad_len), value=0))

    for key in collated:
        collated[key] = torch.stack(collated[key]).long()
    return collated


class MultipleNegativeRankingLoss(nn.Module):
    """Loss to enforce separation between positive and negative pairs"""
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale

    def forward(self, pos_sim, neg_sim):
        scores = torch.stack([pos_sim, neg_sim], dim=1) * self.scale
        labels = torch.zeros(pos_sim.size(0), dtype=torch.long, device=device)
        return F.cross_entropy(scores, labels)


class SemanticMatcher:
    """Trainer for semantic matching"""
    def __init__(self, model_path, adapter_path=None, career_data_path=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = MultiScaleSimilarityModel(model_path, self.tokenizer, adapter_path).to(device)
        self.criterion = MultipleNegativeRankingLoss(scale=20.0)
        self.scaler = GradScaler() if device.type == "cuda" else None
        self.career_data_path = career_data_path
        with open(career_data_path, 'r', encoding='utf-8') as f:
            career_data = json.load(f)
        self.career_texts = [item["desc"] for item in career_data]
        self.career_codes = [item["code"] for item in career_data]
        logger.info("Semantic matcher initialized")

    def train(self, data_path, epochs=10, batch_size=4, val_ratio=0.2, accumulation_steps=4):
        start_time = time.time()
        train_dataset = DynamicTripletDataset(data_path, self.tokenizer, self.career_data_path, mode='train', val_ratio=val_ratio)
        val_dataset = DynamicTripletDataset(data_path, self.tokenizer, self.career_data_path, mode='val', val_ratio=val_ratio)
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=dynamic_collate_fn, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, collate_fn=dynamic_collate_fn, pin_memory=True)

        best_lr, _ = self.auto_lr_search(train_loader, val_loader)
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=best_lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs // accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_f08 = 0.0
        best_epoch = 0
        train_losses = []
        val_metrics = []

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_loader, accumulation_steps)
            train_losses.append(train_loss)
            val_results = self.evaluate(val_loader)
            val_metrics.append(val_results)
            logger.info(f"Train loss: {train_loss:.4f}")
            logger.info(f"Val metrics: Acc: {val_results['accuracy']:.4f}, F0.8: {val_results['f0.8']:.4f}")

            if val_results["f0.8"] > best_f08:
                best_f08 = val_results["f0.8"]
                best_epoch = epoch +6
                self.save_model("best_model")
                logger.info(f"Best model saved with F0.8: {best_f08:.4f}")

        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        self.plot_training(train_losses, val_metrics)
        return best_f08

    def train_epoch(self, data_loader, accumulation_steps):
        self.model.train()
        epoch_loss = 0.0
        total_steps = len(data_loader)
        progress_bar = tqdm(data_loader, desc="Training")

        for step, batch in enumerate(progress_bar):
            patent_ids = batch["patent_ids"].to(device)
            patent_mask = batch["patent_mask"].to(device)
            pdesc_ids = batch["pdesc_ids"].to(device)
            pdesc_mask = batch["pdesc_mask"].to(device)
            ndesc_ids = batch["ndesc_ids"].to(device)
            ndesc_mask = batch["ndesc_mask"].to(device)

            with autocast(enabled=self.scaler is not None):
                pos_sim = self.model(patent_ids, patent_mask, pdesc_ids, pdesc_mask)
                neg_sim = self.model(patent_ids, patent_mask, ndesc_ids, ndesc_mask)
                loss = self.criterion(pos_sim, neg_sim) / accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accumulation_steps == 0 or step == total_steps - 1:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            epoch_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * accumulation_steps})

        return epoch_loss / total_steps

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_pos_sim = []
        all_neg_sim = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                patent_ids = batch["patent_ids"].to(device)
                patent_mask = batch["patent_mask"].to(device)
                pdesc_ids = batch["pdesc_ids"].to(device)
                pdesc_mask = batch["pdesc_mask"].to(device)
                ndesc_ids = batch["ndesc_ids"].to(device)
                ndesc_mask = batch["ndesc_mask"].to(device)

                pos_sim = self.model(patent_ids, patent_mask, pdesc_ids, pdesc_mask)
                neg_sim = self.model(patent_ids, patent_mask, ndesc_ids, ndesc_mask)
                preds = (pos_sim > neg_sim).long().cpu().numpy()
                labels = np.ones_like(preds)

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_pos_sim.extend(pos_sim.cpu().numpy())
                all_neg_sim.extend(neg_sim.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        f0_8 = (1 + 0.8**2) * (precision * recall) / (0.8**2 * precision + recall + 1e-12)
        avg_pos_sim = np.mean(all_pos_sim)
        avg_neg_sim = np.mean(all_neg_sim)
        avg_margin = avg_pos_sim - avg_neg_sim

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f0.8": f0_8,
            "avg_pos_sim": avg_pos_sim,
            "avg_neg_sim": avg_neg_sim,
            "avg_margin": avg_margin
        }

    def auto_lr_search(self, train_loader, val_loader):
        lr_range = [1e-5, 3e-5, 5e-5, 1e-4]
        best_lr = lr_range[0]
        best_f08 = 0.0
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        for lr in lr_range:
            optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.01)
            self.model.train()
            for batch in train_loader:
                patent_ids = batch["patent_ids"].to(device)
                patent_mask = batch["patent_mask"].to(device)
                pdesc_ids = batch["pdesc_ids"].to(device)
                pdesc_mask = batch["pdesc_mask"].to(device)
                ndesc_ids = batch["ndesc_ids"].to(device)
                ndesc_mask = batch["ndesc_mask"].to(device)

                with autocast(enabled=self.scaler is not None):
                    pos_sim = self.model(patent_ids, patent_mask, pdesc_ids, pdesc_mask)
                    neg_sim = self.model(patent_ids, patent_mask, ndesc_ids, ndesc_mask)
                    loss = self.criterion(pos_sim, neg_sim)

                optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            val_results = self.evaluate(val_loader)
            if val_results["f0.8"] > best_f08:
                best_f08 = val_results["f0.8"]
                best_lr = lr

            self.model.load_state_dict(original_state)

        return best_lr, best_f08

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.config.to_dict(),
        }
        torch.save(model_state, os.path.join(output_dir, "model_checkpoint.pth"))
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")

    def plot_training(self, train_losses, val_metrics):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.title("Training Loss")
        plt.legend()

        plt.subplot(2, 1, 2)
        val_f08s = [m["f0.8"] for m in val_metrics]
        val_accs = [m["accuracy"] for m in val_metrics]
        plt.plot(val_f08s, label="Val F0.8", marker="o")
        plt.plot(val_accs, label="Val Accuracy", marker="o")
        plt.title("Validation Metrics")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join("./logs", "training_metrics.png"))
        logger.info("Training metrics plot saved")

    def match_career(self, patent_text: str, threshold: float = 0.5) -> Tuple[Optional[str], Optional[str], float]:
        """Match a patent text to the most relevant career text and code"""
        self.model.eval()
        patent_enc = self.tokenizer(patent_text, truncation=False, padding=False, return_tensors="pt",
                                    add_special_tokens=False)
        patent_ids = patent_enc["input_ids"].to(device)
        patent_mask = patent_enc.get("attention_mask", torch.ones(patent_ids.size(1), dtype=torch.long)).to(device)
        if patent_ids.dim() == 1:
            patent_ids = patent_ids.unsqueeze(0)
        if patent_mask.dim() == 1:
            patent_mask = patent_mask.unsqueeze(0)

        max_sim = -1.0
        best_career_text = None
        best_career_code = None

        with torch.no_grad():
            for career_text, career_code in zip(self.career_texts, self.career_codes):
                career_enc = self.tokenizer(career_text, truncation=False, padding=False, return_tensors="pt",
                                            add_special_tokens=False)
                career_ids = career_enc["input_ids"].to(device)
                career_mask = career_enc.get("attention_mask", torch.ones(career_ids.size(1), dtype=torch.long)).to(
                    device)
                if career_ids.dim() == 1:
                    career_ids = career_ids.unsqueeze(0)
                if career_mask.dim() == 1:
                    career_mask = career_mask.unsqueeze(0)

                sim = self.model(patent_ids, patent_mask, career_ids, career_mask).item()
                if sim > max_sim and sim >= threshold:
                    max_sim = sim
                    best_career_text = career_text
                    best_career_code = career_code

        return best_career_text, best_career_code, max_sim


if __name__ == "__main__":
    BASE_MODEL = "D:/Yufile/pycharm3.4/API/gte-large-zh"
    ADAPTER_PATH = "D:/Yufile/vs/domain_pretrain/checkpoint-20916"
    TRAIN_DATA = "API_train_standard.json"
    CAREER_DATA = "D:/Anaconda/atrain/code/job_output.json"  # Expected format: [{"text": "...", "code": "..."}, ...]

    logger.info("Starting Semantic Matching Fine-tuning")
    matcher = SemanticMatcher(model_path=BASE_MODEL, adapter_path=ADAPTER_PATH, career_data_path=CAREER_DATA)
    best_f08 = matcher.train(data_path=TRAIN_DATA, epochs=15, batch_size=4, val_ratio=0.2, accumulation_steps=4)
    logger.info(f"Training completed. Best F0.8: {best_f08:.4f}")

    # Example inference
    patent_text = "一种用于图像处理的深度学习方法..."
    career_text, career_code, similarity = matcher.match_career(patent_text, threshold=0.5)
    if career_text:
        logger.info(f"Matched career: {career_code} - {career_text[:50]}... (Similarity: {similarity:.4f})")
    else:
        logger.info("No matching career found above threshold")
