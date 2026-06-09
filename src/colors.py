import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def extract_dominant_colors(image: Image.Image, k: int = 3) -> list[str]:
    """
    Extract k dominant colors from a PIL image.
    Returns a list of hex color strings.
    """
    img = image.convert("RGB").resize((150, 150))
    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)

    # Sort by cluster frequency (most dominant first)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    sorted_idx = np.argsort(-counts)

    hex_colors = ["#{:02X}{:02X}{:02X}".format(*centers[i]) for i in sorted_idx]
    return hex_colors
