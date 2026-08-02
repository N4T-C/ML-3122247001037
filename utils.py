# importing packages
try:
    import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, scipy as sp, os
    from PIL import Image
    print("Imported successfully")
except ModuleNotFoundError:
    print("Failed to import!")

 
def eda_tabular(df, target = None):
    print("Shape:", df.shape)
    print("\nColumn types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    
 
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(numeric_cols) > 0:
        print("\nSummary statistics:\n", df.describe())
        df[numeric_cols].hist(figsize=(12, 8), bins=20)
        plt.tight_layout()
        plt.show()
    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        if len(numeric_cols)>8:
            sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()
    if "message" in df.columns:
        df["message_length"] = df["message"].str.len()
        plt.hist(df["message_length"], bins=30)
        plt.title("Message Length Distribution")
        plt.show()
    if target is not None:
        plt.figure(figsize=(5,4))
        df[target].value_counts().sort_index().plot(kind="bar")
        plt.title(f"Class Distribution ({target})")
        plt.xlabel(target)
        plt.ylabel("Count")
        plt.show()
    return numeric_cols,categorical_cols

def eda_images(df, label_col="label", image_size=(28, 28), samples=9):

    print("\nShape:", df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes.value_counts())

    print("\nMissing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    labels = df[label_col]
    pixels = df.drop(columns=[label_col])

    print("\nNumber of Images:", len(df))
    print("Image Size:", image_size)
    print("Number of Classes:", labels.nunique())

    # -----------------------------
    # Class Distribution
    # -----------------------------
    print("\nClass Distribution:")
    print(labels.value_counts().sort_index())

    plt.figure(figsize=(8,4))
    labels.value_counts().sort_index().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Random Sample Images
    # -----------------------------
    plt.figure(figsize=(8,8))

    indices = np.random.choice(len(df), samples, replace=False)

    grid = int(np.ceil(np.sqrt(samples)))

    for i, idx in enumerate(indices):

        plt.subplot(grid, grid, i+1)

        img = pixels.iloc[idx].values.reshape(image_size)

        plt.imshow(img, cmap="gray")
        plt.title(f"Label: {labels.iloc[idx]}")
        plt.axis("off")

    plt.suptitle("Sample Images")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Pixel Intensity Distribution
    # -----------------------------
    plt.figure(figsize=(8,4))
    plt.hist(pixels.values.ravel(), bins=50)
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Average Image
    # -----------------------------
    avg_img = pixels.mean(axis=0).values.reshape(image_size)

    plt.figure(figsize=(5,5))
    plt.imshow(avg_img, cmap="gray")
    plt.title("Average Image")
    plt.axis("off")
    plt.show()

    # -----------------------------
    # Pixel Statistics
    # -----------------------------
    print("\nPixel Statistics")
    print("-------------------------")
    print("Minimum Pixel :", pixels.values.min())
    print("Maximum Pixel :", pixels.values.max())
    print("Mean Pixel    :", round(pixels.values.mean(), 2))
    print("Std Deviation :", round(pixels.values.std(), 2))