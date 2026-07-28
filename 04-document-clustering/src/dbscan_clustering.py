# article_categorization.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import umap
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, silhouette_score
from scipy.optimize import linear_sum_assignment
import random
import numpy as np
import warnings


# ignore warnings for cleaner output
warnings.filterwarnings("ignore")

#download the necessary NLTK resources for preprocessing
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')



def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN  # Default POS

# text preprocessing
def preprocess_text(text, lemmatizer, stopwords_set):
    if pd.isnull(text):
        return ""

    # change to string and reduce to lowercase
    text = str(text).lower()

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords_set]

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatize tokens with POS tags
    lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]

    # remove any empty strings
    lemmatized_tokens = [word for word in lemmatized_tokens if word]

    # join tokens back into a single string
    return ' '.join(lemmatized_tokens)

# initiate stopwords set and lemmatizer, preprocess the text data frame
def preprocess_dataset(df):
    stopwords_set = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    print("\nStarting text preprocessing...")
    df['Preprocessed_Text'] = df['Text'].apply(lambda x: preprocess_text(x, lemmatizer, stopwords_set))
    print("Text preprocessing completed.")

    print("\nSample of preprocessed data:")
    print(df[['Text', 'Preprocessed_Text', 'Category']].head())
    return df

#prepare the tagged documents for the doc2vec model
def prepare_tagged_documents(df):
    df['Tokenized_Text'] = df['Preprocessed_Text'].apply(lambda x: x.split())
    tagged_documents = [TaggedDocument(words=row['Tokenized_Text'], tags=[str(index)]) for index, row in df.iterrows()]
    return tagged_documents

#train the doc2vec model
def train_doc2vec_model(tagged_documents,min_count, seed=42):
    model = Doc2Vec(
        vector_size=100,  # Vector dimensions
        window=5,  # context window size
        min_count=min_count,  # minimum word frequency
        workers=4,
        epochs=100,
        dm=1,
        seed=seed
    )

    model.build_vocab(tagged_documents)
    print(f"Vocabulary size: {len(model.wv)}")

    print("Starting Doc2Vec model training...")
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
    print("Doc2Vec model training completed.")

    return model

#extract the vectors from the doc2vec model
def extract_doc_vectors(doc2vec_model, df):
    doc_vectors = [doc2vec_model.dv[str(index)] for index in df.index]
    doc_vectors = np.array(doc_vectors)
    return doc_vectors

#standardize the document vectors
def scale_vectors(doc_vectors):

    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(doc_vectors)
    return vectors_scaled, scaler

#using umap for dimension reduction to 30 dimensions
def apply_umap(vectors, n_components=30, seed=42):

    print("\nStarting UMAP dimensionality reduction...")
    umap_reducer = umap.UMAP(
        n_neighbors=15,
        n_components=n_components,
        metric='cosine',
        random_state=seed
    )
    vectors_umap = umap_reducer.fit_transform(vectors)
    print("UMAP dimensionality reduction completed.")
    return vectors_umap, umap_reducer

#plot the k distance diagram to help to choose the parameters for DBSCAN
def plot_k_distance(vectors_umap,file, k=5):
    print(f"\nPlotting k-distance graph for {file}...")
    neighbors = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(vectors_umap)
    distances, indices = neighbors.kneighbors(vectors_umap)
    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.xlabel('Number of Data Points')
    plt.ylabel(f'{k} Nearest Neighbor Distance')
    plt.title(f'k-Distance Graph (k={k})')
    plt.grid(True)
    plt.show()


# perform dbscan algorithm
def perform_dbscan(vectors_umap, eps, min_samples):
    print("\nPerforming DBSCAN clustering with best parameters...")
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='euclidean',
        n_jobs=-1
    )
    labels = dbscan.fit_predict(vectors_umap)
    print("DBSCAN clustering completed.")
    return labels, dbscan

#maps DBSCAN cluster labels to the actual class labels using the Hungarian algorithm
def map_clusters_to_classes(df, clusters, true_labels, num_classes):

    print("\nMapping clusters to actual classes using the Hungarian algorithm...")

    #create confusion matrix
    conf_matrix = confusion_matrix(true_labels, clusters)

    #determine the size for padding
    num_clusters = conf_matrix.shape[1]
    size = max(num_clusters, num_classes)
    conf_matrix_padded = np.zeros((size, size))
    conf_matrix_padded[:conf_matrix.shape[0], :conf_matrix.shape[1]] = conf_matrix

    # apply Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(-conf_matrix_padded)  #maximizing matches

    # create a mapping from cluster to class
    cluster_to_class = {}
    for i, j in zip(row_ind, col_ind):
        cluster_to_class[j] = i

    # map clusters to classes
    mapped_clusters = np.copy(clusters)
    for cluster_label, class_label in cluster_to_class.items():
        if cluster_label != -1:  # do not map noise
            mapped_clusters[clusters == cluster_label] = class_label

    # assign noise points to a new class
    noise_class = num_classes  # New class index for noise
    mapped_clusters[clusters == -1] = noise_class

    # update LabelEncoder classes to include 'Noise' if necessary
    le = LabelEncoder()
    le.fit(df['Category'])
    class_names = list(le.classes_)
    class_names.append('Noise')

    # map cluster labels to category names
    df['Mapped_Cluster'] = mapped_clusters
    df['Cluster_Category'] = df['Mapped_Cluster'].apply(
        lambda x: class_names[x] if x < num_classes else 'Noise'
    )

    print("\nCluster to Category Mapping:")
    print(df['Cluster_Category'].value_counts())

    return mapped_clusters, class_names

#printing the results
def evaluate_clustering(y_true, y_pred):
    print("\nCalculating Precision, Recall, and F1 Score...")
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {'precision': precision, 'recall': recall, 'f1_score': f1}

#visualize the clustering using umap
def visualize_3d_umap(df, class_names, seed=42):
    print("\nCreating 3D UMAP visualization...")

    # Initialize UMAP for 3D reduction
    umap_3d = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=seed
    )

    # Perform 3D UMAP
    umap_embedding = umap_3d.fit_transform(df[['UMAP1', 'UMAP2', 'UMAP3']].values)

    # Add UMAP3 dimension to DataFrame
    df['UMAP3_3D'] = umap_embedding[:, 2]

    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        df['UMAP1'],
        df['UMAP2'],
        df['UMAP3_3D'],
        c=df['Cluster_Category'].astype('category').cat.codes,
        cmap='tab10',
        alpha=0.7
    )

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Cluster Categories')


    # Set titles and labels
    ax.set_title('UMAP Visualization of DBSCAN Clusters')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_zlabel('UMAP Dimension 3')

    plt.tight_layout()
    plt.show()

# visualization for confusion matrix
def visualize_confusion_matrix(y_true, y_pred, class_labels):

    print("\nVisualizing Confusion Matrix...")
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels[:-1])  #exclude 'Noise' from actual classes
    plt.xlabel('Predicted Cluster')
    plt.ylabel('Actual Category')
    plt.title('Confusion Matrix After Mapping Clusters to Categories')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def main():

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    #read the respective dataset and adjust the df for different set
    for file in ["article_categorization.csv","movie_review_prediction_3000.csv","news_data.csv"]:
        if file == "article_categorization.csv":
            df = pd.read_csv(file)
            best_eps = 0.53
            best_min_samples = 5

        elif file == "movie_review_prediction_3000.csv":
            df = pd.read_csv(file)
            df.columns = ['Text', 'Category']
            best_eps = 0.86
            best_min_samples = 31

        elif file == "news_data.csv":
            df = pd.read_csv(file)
            df = df.drop('Title', axis=1)
            df.columns = ['Text', 'Category']
            best_eps = 0.57
            best_min_samples = 5

        print(f"\n Processing {file}:\n")

        download_nltk_resources()

        df = preprocess_dataset(df)

        tagged_documents = prepare_tagged_documents(df)

        if file == "article_categorization.csv":
            min_count = 5
            doc2vec_model = train_doc2vec_model(tagged_documents,min_count, seed=seed)
        else:
            min_count = 2
            doc2vec_model = train_doc2vec_model(tagged_documents,min_count, seed=seed)

        doc_vectors = extract_doc_vectors(doc2vec_model, df)

        vectors_scaled, scaler = scale_vectors(doc_vectors)

        vectors_umap, umap_reducer = apply_umap(vectors_scaled, n_components=30, seed=seed)

        plot_k_distance(vectors_umap,file=file, k=5)

        clusters, dbscan_model = perform_dbscan(vectors_umap, best_eps, best_min_samples)
        df['Cluster'] = clusters
        print(f"\nCluster distribution for {file}:")
        print(df['Cluster'].value_counts())

        le = LabelEncoder()
        df['Category_Encoded'] = le.fit_transform(df['Category'])
        true_labels = df['Category_Encoded'].values
        num_classes = len(le.classes_)

        mapped_clusters, class_names = map_clusters_to_classes(df, clusters, true_labels, num_classes)

        #print out the result
        evaluation_metrics = evaluate_clustering(true_labels, mapped_clusters)

        # re-apply UMAP with 3 dimension on the scaled vectors for visualization
        print(f"\nStarting 3D UMAP dimensionality reduction for visualization of {file}...")
        umap_3d = umap.UMAP(
            n_neighbors=15,
            n_components=3,
            metric='cosine',
            random_state=seed
        )
        vectors_umap_3d = umap_3d.fit_transform(vectors_scaled)
        df['UMAP1'] = vectors_umap_3d[:, 0]
        df['UMAP2'] = vectors_umap_3d[:, 1]
        df['UMAP3'] = vectors_umap_3d[:, 2]
        print("3D UMAP dimensionality reduction completed.")

        visualize_3d_umap(df, class_names, seed=seed)

        visualize_confusion_matrix(true_labels, mapped_clusters, class_names)


if __name__ == "__main__":
    main()
