# diagnose_reads.py
import os, cv2
root = r'.\\imagedataset'
categories = ['spawning','spent']
bad = []
ok = []
for c in categories:
    path = os.path.join(root, c)
    for fname in sorted(os.listdir(path)):
        p = os.path.join(path, fname)
        if not os.path.isfile(p):
            bad.append((p,'not a file')); continue
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            bad.append((p,'imread returned None')); continue
        try:
            # quick resize test
            img2 = cv2.resize(img, (256,256))
        except Exception as e:
            bad.append((p, f'resize failed: {e}')); continue
        ok.append(p)
print('OK count:', len(ok))
print('Bad count:', len(bad))
for b in bad[:50]:
    print('BAD:', b)