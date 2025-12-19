from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import ColorExtract

TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PATH = TEST_DATA_DIR / "sample001_rect.jpg"

def run_test():
  # テスト用画像パスの解決
  image = cv2.imread(SAMPLE_PATH)
  if image is None:
    print(f"Error: Image not found at {SAMPLE_PATH}")
    return

  print("--- Running ColorExtract Test ---")

  # 1. CLAHE テスト
  ce = ColorExtract.Clahe(clip_limit=2.0, tile_grid_size=(8, 8))
  enhanced_img = ce.apply_yuv(image)
  print("CLAHE processing: Done.")

  # 2. KMeans テスト
  cfe_km = ColorExtract.ColorFeatureExtractor_KMeans(enhanced_img, n_clusters=3)
  cfe_km.warmup()
  dominant_colors = cfe_km.get_dominant_colors()
  print(f"KMeans Extraction: Found {len(dominant_colors)} colors.")

  # 3. Color Matcher テスト
  matcher = ColorExtract.ColorMatcher
  for color in dominant_colors:
    en_name = matcher.match_detailed(color, input_format="rgb")
    jp_name = matcher.to_japanese(en_name)
    print(f"Matched: {en_name} ({jp_name}) - RGB: {color}")

  # 結果の可視化
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  plt.title("Original")
  
  plt.subplot(1, 2, 2)
  plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
  plt.title("Enhanced (CLAHE)")
  
  plt.show()

if __name__ == '__main__':
  run_test()