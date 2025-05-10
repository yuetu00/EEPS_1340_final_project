import numpy as np
from PIL import Image
from transformers import pipeline
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans



# Before using the model download all the data linked in the README and follow the comments as you use the code
# NOTE: You should only need to download some packages, data, and fill out the variables for data_path no further coding is necessary to run the preprocessing 
data_path_training = ["Data/ejecta_train", # Paths for ejecta, none, and old  for train, validate, and test [replace with your own filepath]
                      "Data/oldcrater_train",
                      "Data/none_train1"]
data_path_train_save = "Dataframe/all_train.csv" # Paths for saving csv for ejecta, none, and old  for train, validate, and test [replace with your own filepath]

data_path_validating = ["Data/val/ejecta",
                        "Data/val/oldcrater",
                        "Data/val/none"]
data_path_validate_save = "Dataframe/all_validate.csv"

data_path_testing = ["Data/test/ejecta",
                     "Data/val/oldcrater",
                     "Data/val/none"]
data_path_test_save = "Dataframe/all_test.csv"

# Create a pipeline for extracting features from Google's Vision Transformer (ViT) model
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-384", pool=True)

def img_to_dataframe(data_dir,mode):
    #   Check if mode is training, validating, or testing
    accept_modes = ["train", "validate","test"]
    if mode not in accept_modes:
        print("inputted mode is not an accepted: train, validate, test ")
        return None

    # Lists all files in the directory
    file_ejecta = os.listdir(data_dir[0])
    file_old = os.listdir(data_dir[1])
    file_none = os.listdir(data_dir[2])

    # loads each image into a list
    images_ejecta = [Image.open(f"{data_dir[0]}/{file}") for file in file_ejecta if file.endswith('.jpg')]  # Open multiple images
    images_old = [Image.open(f"{data_dir[1]}/{file}") for file in file_old if file.endswith('.jpg')]
    images_none = [Image.open(f"{data_dir[2]}/{file}") for file in file_none if file.endswith('.jpg')]

    # Extracts features from the image(s) for train, test, and validation
    features_ejecta= pipe(images_ejecta)
    features_old= pipe(images_old)  # Extract features
    features_none= pipe(images_none)

    print(np.array(features_ejecta).squeeze().shape) # Output: (2, 1, 768) or (num_images, num_rows, num_cols)
    print(np.array(features_old).squeeze().shape)
    print(np.array(features_none).squeeze().shape)

    ejecta = pd.DataFrame(np.array(features_ejecta).squeeze()) # Convert array to DataFrames
    ejecta['classification'] = 'ejecta'

    old = pd.DataFrame(np.array(features_old).squeeze())
    old['classification'] = 'old_crater'

    none = pd.DataFrame(np.array(features_none).squeeze())
    none['classification'] = 'none' 
    
    # concats ejecta, old, and none together for train, test, and validate
    if mode  == "train": 
        all_train_data = pd.concat([ejecta, old, none])
        all_train_data.to_csv(data_path_train_save)
    elif mode == "validate":
        all_validate_data = pd.concat([ejecta, old, none])
        all_validate_data.to_csv(data_path_validate_save)
    elif mode == "test":
        all_test_data = pd.concat([ejecta, old, none])
        all_test_data.to_csv(data_path_test_save)

""" Expected Sizes of CSV Files:
#  All_train: (1373 x 768)
#  All_validate: (780 x 768)
#  All_test: (780 x 768) 
"""

# uncomment the calls one ata time until you've ran them all
#img_to_dataframe(data_path_training,"train") 
#img_to_dataframe(data_path_validating,"validate") #613
#img_to_dataframe(data_path_testing,"test") #647



# Current Things To Do: 
# make a plot
# All the data together in a PCA 1PCA vs 2PCA and classify based on crater or not crater
# Mean and Standard deviation based on different x vs y 
# Final Touches: convert to 0 and 1s for crater or not crater, standardize the data. 
#train['classification'] = train['classification'] == 'ejecta' or train['classification'] == 'old_crater'


#plot PCA for visualization 
#load the combined data 
df = pd.read_csv(data_path_train_save)

#drop columns that are not features 
features = df.drop(columns=['classification', 'Unnamed: 0'])

#pca
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.1%}")

#plot 
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]


plt.figure(figsize=(8, 6))
for label in df['classification'].unique():
    subset = df[df['classification'] == label]
    plt.scatter(subset['pca1'], subset['pca2'], label=label, alpha=0.6)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of ViT Features")
plt.legend()
plt.grid(True)
plt.show()


#binary classification crater vs non crater
df['binary_class'] = df['classification'].isin(['ejecta', 'old_crater']).astype(int)

#standardize our features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


#t-SNE for 3-class visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(features_scaled)

df['tsne1'] = tsne_result[:, 0]
df['tsne2'] = tsne_result[:, 1]

# Use the original multiclass labels
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='tsne1', y='tsne2', hue='classification', palette='Set1', alpha=0.6)
plt.title("t-SNE of ViT Features - 3-Class (Ejecta, Old Crater, None)")
plt.grid(True)
plt.show()

#kmeans
kmeans = KMeans(n_clusters=2, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(features_scaled)

# Plot using PCA or t-SNE coordinates
sns.scatterplot(data=df, x='pca1', y='pca2', hue='kmeans_cluster', palette='Set2', alpha=0.6)
plt.title("K-Means Clustering on PCA")
plt.grid(True)
plt.show()


