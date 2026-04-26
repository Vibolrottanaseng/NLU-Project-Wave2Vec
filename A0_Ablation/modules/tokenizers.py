import json
import re
from collections import Counter


class Tokenizer(object):
    def __init__(self, args):
        self.ann = json.load(open(args.ann_path, 'r'))
        self.threshold = args.threshold

        # ✅ SPECIAL TOKENS
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.token2idx, self.idx2token = self.create_vocabulary()

        self.pad_idx = self.token2idx[self.pad_token]
        self.bos_idx = self.token2idx[self.bos_token]
        self.eos_idx = self.token2idx[self.eos_token]

    def create_vocabulary(self):
        total_tokens = []

        for ex in self.ann['train']:
            tokens = self.clean_report(ex['report']).split()
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)

        vocab = [k for k, v in counter.items() if v >= self.threshold]

        vocab = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token
        ] + sorted(vocab)

        token2idx = {tok: i for i, tok in enumerate(vocab)}
        idx2token = {i: tok for tok, i in token2idx.items()}

        return token2idx, idx2token

    def get_vocab_size(self):
        return len(self.token2idx)

    def get_id_by_token(self, token):
        return self.token2idx.get(token, self.token2idx[self.unk_token])

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = [self.bos_idx]

        for t in tokens:
            ids.append(self.get_id_by_token(t))

        ids.append(self.eos_idx)
        return ids

    def decode(self, ids):
        words = []
        for idx in ids:
            if idx == self.eos_idx:
                break
            if idx > 0:
                words.append(self.idx2token.get(idx, self.unk_token))
        return " ".join(words)

    def decode_batch(self, batch):
        return [self.decode(ids) for ids in batch]

    def clean_report(self, report):
        report = report.lower()
        report = re.sub(r"[.,?;*!%^&_+():\-\[\]{}]", "", report)
        return report