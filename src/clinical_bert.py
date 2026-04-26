from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class ClinicalBERTEncoder:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    @torch.no_grad()
    def encode_batch(self, texts, batch_size=16):
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256
            )

            if torch.cuda.is_available():
                tokens = {k: v.cuda() for k, v in tokens.items()}

            output = self.model(**tokens)

            # mean pooling
            last_hidden = output.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)

            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)

            emb = summed / counts
            emb = emb.cpu().numpy()

            embeddings.append(emb)

        return np.vstack(embeddings)
    
class CaseEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.cache = {}

    def hash(self, text):
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()

    def encode_case(self, current, priors):
        texts = [current] + priors

        cached = []
        to_run = []
        idx_map = []

        for i, t in enumerate(texts):
            key = self.hash(t)
            if key in self.cache:
                cached.append((i, self.cache[key]))
            else:
                to_run.append(t)
                idx_map.append(i)

        new_embs = self.encoder.encode_batch(to_run) if to_run else []

        for t, emb, i in zip(to_run, new_embs, idx_map):
            self.cache[self.hash(t)] = emb
            cached.append((i, emb))

        cached.sort(key=lambda x: x[0])
        return np.stack([e for _, e in cached])