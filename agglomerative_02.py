from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from features_00 import load_and_extract_features
import os, cv2, re
import numpy as np

images, names, X = load_and_extract_features()

model = AgglomerativeClustering(n_clusters=2, linkage="ward")
labels = model.fit_predict(X)

sil = silhouette_score(X, labels)
print(f"Agglomerative Silhouette: {sil:.4f}")

for cid in np.unique(labels):
    path = f"agglo_clusters/cluster_{cid}"
    os.makedirs(path, exist_ok=True)
    for img, lbl, fname in zip(images, labels, names):
        if lbl != cid:
            continue

        # Build tokens robustly: split on OS separators or common flattening delimiters
        if os.sep in fname or ("/" in fname) or ("\\" in fname):
            comps = [c for c in re.split(r"[\\/]+", fname) if c]
        else:
            comps = [c for c in re.split(r"[_\- ]+", fname) if c]

        # Find first occurrence of Male or Female (case-insensitive)
        idx = None
        for i, c in enumerate(comps):
            if c.lower() in ("male", "female"):
                idx = i
                break
        if idx is None:
            for i, c in enumerate(comps):
                if "male" in c.lower() or "female" in c.lower():
                    idx = i
                    break
        if idx is None:
            continue

        sub_comps = comps[idx:]
        last = sub_comps[-1]
        name_no_ext = os.path.splitext(last)[0]
        sub_comps[-1] = name_no_ext
        save_name = "_".join(sub_comps).replace(" ", "_")
        save_path = os.path.join(path, f"{save_name}.jpg")

        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            bgr = img
        cv2.imwrite(save_path, bgr)


def report_gonadal_stage_counts(names, labels):
    """Print counts of gonadal-stage labels per cluster and percentages of total images.

    This inspects path components and detects 'Mature', 'Spawning', and 'Spent'.
    """
    unique_ids = sorted(set(labels))
    for cluster_id in unique_ids:
        mature_count = 0
        spawning_count = 0
        spent_count = 0
        total_in_cluster = 0

        for lbl, fname in zip(labels, names):
            if lbl != cluster_id:
                continue
            total_in_cluster += 1

            norm = os.path.normpath(fname).lstrip(os.sep).lstrip("./\\")
            if os.sep in fname or ("/" in fname) or ("\\" in fname):
                tokens = [c for c in re.split(r"[\\/]+", norm) if c]
            else:
                tokens = [c for c in re.split(r"[_\- ]+", norm) if c]

            tokens_lower = [t.lower() for t in tokens]
            if any("mature" == t or "mature" in t for t in tokens_lower):
                mature_count += 1
            if any("spawning" == t or "spawning" in t for t in tokens_lower):
                spawning_count += 1
            if any("spent" == t or "spent" in t for t in tokens_lower):
                spent_count += 1

        if total_in_cluster > 0:
            pct_mature = (mature_count / total_in_cluster) * 100.0
            pct_spawning = (spawning_count / total_in_cluster) * 100.0
            pct_spent = (spent_count / total_in_cluster) * 100.0
        else:
            pct_mature = pct_spawning = pct_spent = 0.0

        print(f"\nCluster {cluster_id}:")
        print(f"  Total images in cluster: {total_in_cluster}")
        print(f"  Mature: {mature_count} ({pct_mature:.2f}%)")
        print(f"  Spawning: {spawning_count} ({pct_spawning:.2f}%)")
        print(f"  Spent: {spent_count} ({pct_spent:.2f}%)")


# Print the report
report_gonadal_stage_counts(names, labels)
