# Depth download (Phase 0 / Task 0b)

- Source: `gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/<dish>/depth_raw.png`, accessible via plain HTTPS at `https://storage.googleapis.com/nutrition5k_dataset/...`. No auth required (public bucket).
- Method: `scripts/download_depth.sh` — `xargs -P 16` parallel `curl`. Wall-clock = **35s** for 3490 dishes (no rate-limit hit).
- Result: **3489/3490 valid 16-bit (480,640) depth_raw.png** files. One dish (`dish_1564159636`) has a 0-byte object in GCS (`x-goog-stored-content-length: 0`, MD5 = empty-string MD5). This is a dataset-level corruption — the file simply doesn't exist on Google's side, so it's been removed from local disk and will be filtered by `Nutrition5kRGBD(require_depth=True)`.
- Effective dish counts now (with depth):
  - rgb_train_ids ∩ available ∩ has_depth = ~2754 (one fewer if dish_1564159636 was in train; verify in G1)
  - rgb_test_ids ∩ available ∩ has_depth = ~507 (verify)

## Reproduce

```bash
bash scripts/download_depth.sh data/sample/available_dish_ids.txt data/sample/imagery 16
```
