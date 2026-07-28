import os
os.environ['OMP_NUM_THREADS'] = '15'
import pandas as pd
import nltk
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')

# 数据预处理函数
def get_data(filename, text_col="Text", category_col="Category", range_xticks=400):
    data = pd.read_csv(filename)
    os.makedirs("./output", exist_ok=True)
    data.rename(columns={text_col: "Text", category_col: "Category"}, inplace=True)
    return data[["Text", "Category"]]

# 文本清洗和特征提取
def get_data_feature(data, vector_size=100, min_count=1, epochs=100):
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)

    data['Cleaned_Text'] = data['Text'].apply(preprocess_text)
    documents = data['Cleaned_Text'].tolist()
    tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(documents)]

    model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    X = [model.dv[str(i)] for i in range(len(tagged_data))]
    return X, data

# 聚类算法实现
def gmm_clustering(X, data):
    num_clusters = data["Category"].nunique()
    X = np.array(X)
    gmm = GaussianMixture(n_components=num_clusters, random_state=0)
    return gmm.fit_predict(X)

# 映射聚类标签到真实标签
def map_clusters_to_true_labels(cluster_labels, true_labels):
    cluster_map = {}
    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        true_labels_in_cluster = true_labels[cluster_indices]
        label_counts = Counter(true_labels_in_cluster)
        most_common_label = label_counts.most_common(1)[0][0]
        cluster_map[cluster] = most_common_label
    return cluster_map

# 打印结果
# 打印结果并绘制混淆矩阵
def print_results(cluster_labels, true_labels):
    cluster_map = map_clusters_to_true_labels(cluster_labels, true_labels)
    cluster_labels = np.array([cluster_map[label] for label in cluster_labels])

    precision = precision_score(true_labels, cluster_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, cluster_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, cluster_labels, average='macro', zero_division=0)

    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}\n')

    # 计算并绘制混淆矩阵
    cm = confusion_matrix(true_labels, cluster_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# PCA降维和可视化
def pca_visualization(X, labels):
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=labels, cmap='viridis')

    plt.colorbar(scatter)
    ax.set_title('PCA 3D Visualization of Text Features')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    plt.show()

if __name__ == "__main__":
    for filename in ["./dataset/dataset1.csv", "./dataset/dataset2.csv", "./dataset/dataset3.csv"]:
        if filename == "./dataset/dataset1.csv":
            df = get_data(filename, text_col="Text", category_col="Category")
        elif filename == "./dataset/dataset2.csv":
            df = get_data(filename, text_col="Excerpt", category_col="Category")
        elif filename == "./dataset/dataset3.csv":
            df = get_data(filename, text_col="review", category_col="sentiment")

        X, df2 = get_data_feature(df)
        pred_result = gmm_clustering(X, df2)

        print(f'Datasets: {filename}')
        print_results(pred_result, df2["Category"])

        # 可视化PCA降维
        pca_visualization(X, pred_result)
