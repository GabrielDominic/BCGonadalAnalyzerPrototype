from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from features_00 import load_and_extract_features
import os, cv2, re, shutil

images, names, X = load_and_extract_features()

k = 3  # choose any value
model = KMeans(n_clusters=k, random_state=42)
labels = model.fit_predict(X)

sil = silhouette_score(X, labels)
print(f"K-Means Silhouette: {sil:.4f}")

# Ensure a clean output folder: remove existing `kmeans_clusters` and recreate
outroot = "kmeans_clusters"
if os.path.exists(outroot):
    try:
        shutil.rmtree(outroot)
    except Exception as e:
        print(f"Warning: couldn't remove existing '{outroot}': {e}")
os.makedirs(outroot, exist_ok=True)

# Save results
for cluster_id in range(k):
    path = f"kmeans_clusters/cluster_{cluster_id}"
    os.makedirs(path, exist_ok=True)

    for img, lbl, fname in zip(images, labels, names):
        if lbl != cluster_id:
            continue

        # Build tokens robustly: split on OS separators or common flattening delimiters
        if os.sep in fname or ("/" in fname) or ("\\" in fname):
            comps = [c for c in re.split(r"[\\/]+", fname) if c]
        else:
            # likely a flattened filename using underscores or other separators
            comps = [c for c in re.split(r"[_\- ]+", fname) if c]

        # Find first occurrence of Male or Female (case-insensitive)
        idx = None
        for i, c in enumerate(comps):
            if c.lower() in ("male", "female"):
                idx = i
                break
        if idx is None:
            # Try searching tokens for 'Male'/'Female' as substrings
            for i, c in enumerate(comps):
                if "male" in c.lower() or "female" in c.lower():
                    idx = i
                    break
        if idx is None:
            # If still not found, skip saving this file
            continue

        # Build save filename starting at Male/Female component
        sub_comps = comps[idx:]
        # Remove extension from last component if present
        last = sub_comps[-1]
        name_no_ext = os.path.splitext(last)[0]
        sub_comps[-1] = name_no_ext
        # Join with underscores and sanitize spaces
        save_name = "_".join(sub_comps).replace(" ", "_")
        save_path = os.path.join(path, f"{save_name}.jpg")

        # Convert color and save
        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            bgr = img
        cv2.imwrite(save_path, bgr)


def report_gonadal_stage_counts(names, labels, k):
    """Print counts of Gonadal Stage images per cluster and the percentage of each stage .

    The function inspects path components in `names` and looks for any
    component equal to its Gonadal Stages (case-insensitive).
    """
    for cluster_id in range(k):
        mature_count = 0
        spawning_count = 0
        spent_count = 0
        total_in_cluster = 0

        for lbl, fname in zip(labels, names):
            if lbl != cluster_id:
                continue
            total_in_cluster += 1

            # Tokenize the filename/path robustly to handle both normal paths
            # and flattened underscore-style filenames like your example.
            norm = os.path.normpath(fname).lstrip(os.sep).lstrip("./\\")
            if os.sep in fname or ("/" in fname) or ("\\" in fname):
                tokens = [c for c in re.split(r"[\\/]+", norm) if c]
            else:
                tokens = [c for c in re.split(r"[_\- ]+", norm) if c]

            tokens_lower = [t.lower() for t in tokens]
            # Detect presence of Gonadal Stages either as exact token
            # or as substring inside a flattened token. If tokenization
            # doesn't find matches, fall back to checking the whole
            # filename/path string for stage substrings.
            found_mature = any("mature" == t or "mature" in t for t in tokens_lower)
            found_spawning = any("spawning" == t or "spawning" in t for t in tokens_lower)
            found_spent = any("spent" == t or "spent" in t for t in tokens_lower)

            if not (found_mature or found_spawning or found_spent):
                lowers = fname.lower()
                found_mature = "mature" in lowers
                found_spawning = "spawning" in lowers
                found_spent = "spent" in lowers

            if found_mature:
                mature_count += 1
            if found_spawning:
                spawning_count += 1
            if found_spent:
                spent_count += 1

        # Percentages relative to the total images in the cluster
        if total_in_cluster > 0:
            pct_mature = (mature_count / total_in_cluster) * 100.0
            pct_spawning = (spawning_count / total_in_cluster) * 100.0
            pct_spent = (spent_count / total_in_cluster) * 100.0
            pct_mature_str = f"{pct_mature:.2f}%"
            pct_spawning_str = f"{pct_spawning:.2f}%"
            pct_spent_str = f"{pct_spent:.2f}%"
        else:
            pct_mature_str = "N/A (no images)"
            pct_spawning_str = "N/A (no images)"
            pct_spent_str = "N/A (no images)"

        print(f"\nCluster {cluster_id}:")
        print(f"  Total images in cluster: {total_in_cluster}")
        print(f"  Mature: {mature_count} ({pct_mature_str})")
        print(f"  Spawning: {spawning_count} ({pct_spawning_str})")
        print(f"  Spent: {spent_count} ({pct_spent_str})")

# Print the report
report_gonadal_stage_counts(names, labels, k)
