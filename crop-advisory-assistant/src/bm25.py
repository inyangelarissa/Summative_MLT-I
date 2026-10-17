"""Minimal BM25 (Okapi) implementation. Used for retrieval context and as baseline B1."""
import math
import re
from collections import Counter, defaultdict

STOP = set("a an the of and or to in on for with is are was were be been it its this that these those at by from as "
           "how what when why which who do does can i my me we you your should would could will there their than then "
           "if not no yes about into over after before".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.dl = [sum(c.values()) for c in self.tf]
        self.avgdl = sum(self.dl) / max(1, len(self.dl))
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def top_k(self, query, k=3):
        scores = defaultdict(float)
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += self.idf[t] * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [i for i, _ in ranked]
