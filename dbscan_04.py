from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from features_00 import load_and_extract_features
import os, cv2, re
import numpy as np

images, names, X = load_and_extract_features()

model = DBSCAN(eps=3, min_samples=3)
labels = model.fit_predict(X)

valid = labels != -1

if len(np.unique(labels[valid])) > 1:
    sil = silhouette_score(X[valid], labels[valid])
    print(f"DBSCAN Silhouette: {sil:.4f}")
else:
    print("DBSCAN: Not enough clusters.")

clusters = sorted(list(set(labels) - {-1}))

for cid in clusters:
    path = f"dbscan_clusters/cluster_{cid}"
    os.makedirs(path, exist_ok=True)
    for img, lbl, fname in zip(images, labels, names):
        if lbl != cid:
            continue

        if os.sep in fname or ("/" in fname) or ("\\" in fname):
            comps = [c for c in re.split(r"[\\/]+", fname) if c]
        else:
            comps = [c for c in re.split(r"[_\- ]+", fname) if c]

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

os.makedirs("dbscan_clusters/noise", exist_ok=True)
for img, lbl, fname in zip(images, labels, names):
    if lbl == -1:
        if os.sep in fname or ("/" in fname) or ("\\" in fname):
            comps = [c for c in re.split(r"[\\/]+", fname) if c]
        else:
            comps = [c for c in re.split(r"[_\- ]+", fname) if c]

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
            save_name = os.path.splitext(os.path.basename(fname))[0]
        else:
            sub_comps = comps[idx:]
            last = sub_comps[-1]
            sub_comps[-1] = os.path.splitext(last)[0]
            save_name = "_".join(sub_comps).replace(" ", "_")

        save_path = os.path.join("dbscan_clusters/noise", f"{save_name}.jpg")
        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            bgr = img
        cv2.imwrite(save_path, bgr)


def report_gonadal_stage_counts(names, labels):
    unique_ids = sorted(set(labels))
    for cluster_id in unique_ids:
        mature_count = spawning_count = spent_count = total_in_cluster = 0
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


report_gonadal_stage_counts(names, labels)
