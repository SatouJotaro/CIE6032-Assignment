from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1. 准备样本数据
documents = [
    "The cat in the hat.",
    "A cat is a fine pet.",
    "Dogs and cats are animals.",
    "I love my pet dog.",
    "Computers are silicon-based machines."
]

# 2. 构建 TF-IDF 矩阵
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# 3. 使用 SVD 降维（设定潜在主题数为 2）
lsa_model = TruncatedSVD(n_components=2)
lsa_topic_matrix = lsa_model.fit_transform(tfidf_matrix)

# 查看降维后的结果
print("降维后的文档坐标：\n", lsa_topic_matrix)