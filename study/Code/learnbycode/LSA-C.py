import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ── 1. 准备文档 ──────────────────────────────────────────
documents = [
    "The cat in the hat.",
    "A cat is a fine pet.",
    "Dogs and cats are animals.",
    "I love my pet dog.",
    "Computers are silicon-based machines.",
    "I enjoy programming computers.",
]
doc_names = [f"Doc{i+1}" for i in range(len(documents))]

# ── 2. 构建 TF-IDF 矩阵 ──────────────────────────────────
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()

print(f"词汇表大小: {len(vocab)}")
print(f"TF-IDF 矩阵形状 (词数 × 文档数): {tfidf_matrix.T.shape}\n")

# ── 3. SVD 降维：保留 k=2 个潜在主题 ────────────────────
# k 的选择：越小越抽象，越大越细节，通常取 50~300（此处取 2 方便可视化）
k = 2
lsa = TruncatedSVD(n_components=k, random_state=42)
doc_topic_matrix = lsa.fit_transform(tfidf_matrix)  # 文档在主题空间中的坐标

# ── 4. 查看每个主题最相关的词 ────────────────────────────
print("=== 每个潜在主题的 Top 词 ===")
for i, component in enumerate(lsa.components_):
    top_indices = component.argsort()[-5:][::-1]  # 取权重最大的 5 个词
    top_words = [vocab[j] for j in top_indices]
    print(f"  主题 {i+1}: {', '.join(top_words)}")

# ── 5. 查看文档在主题空间的坐标 ─────────────────────────
print("\n=== 文档在潜在主题空间的坐标 ===")
df = pd.DataFrame(
    doc_topic_matrix,
    index=doc_names,
    columns=[f"主题{i+1}" for i in range(k)]
)
print(df.round(3))

# ── 6. 计算文档间的语义相似度 ────────────────────────────
print("\n=== 文档语义相似度矩阵（余弦相似度）===")
similarity = cosine_similarity(doc_topic_matrix)
sim_df = pd.DataFrame(similarity, index=doc_names, columns=doc_names)
print(sim_df.round(3))

# ── 7. 查询一个新句子与哪篇文档最相似 ───────────────────
query = ["I have a dog as my pet."]
query_tfidf = vectorizer.transform(query)       # 用已有词汇表转换
query_topic = lsa.transform(query_tfidf)        # 投影到潜在语义空间

scores = cosine_similarity(query_topic, doc_topic_matrix)[0]
ranked = np.argsort(scores)[::-1]

print(f"\n=== 查询句: '{query[0]}' ===")
for idx in ranked:
    print(f"  {doc_names[idx]}: {scores[idx]:.3f}  ← {documents[idx]}")