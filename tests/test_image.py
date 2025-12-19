from pathlib import Path
import cv2
import numpy as np
import ColorExtract

TEST_DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PATH = TEST_DATA_DIR / "sample001_rect.jpg"

def run_test():
  # テスト用画像パスの解決
  image = cv2.imread(SAMPLE_PATH)
  if image is None:
    print(f"Error: Image not found at {SAMPLE_PATH}")
    return

  print("--- Running ColorExtract Test (OpenCV Display) ---")

  # 1. CLAHE テスト（コントラスト補正）
  ce = ColorExtract.Clahe(clip_limit=2.0, tile_grid_size=(8, 8))
  enhanced_img = ce.apply_yuv(image)
  print("CLAHE processing: Done.")

  # 2. KMeans テスト（代表色抽出）
  n_colors = 3
  cfe_km = ColorExtract.ColorFeatureExtractor_KMeans(enhanced_img, n_clusters=n_colors)
  cfe_km.warmup()
  dominant_colors = cfe_km.get_dominant_colors() # RGBで返る
  ratios = cfe_km.get_ratios()
  print(f"KMeans Extraction: Found {len(dominant_colors)} colors.")

  # 3. Color Matcher テスト & カラーパレット画像の生成
  # 結果を可視化するためのバーを作成 (高さ100px, 幅は画像の横幅に合わせる)
  palette_height = 100
  palette = np.zeros((palette_height, enhanced_img.shape[1], 3), dtype=np.uint8)
  
  current_x = 0
  for i in range(len(dominant_colors)):
    color_rgb = dominant_colors[i]
    ratio = ratios[i]
    
    # 色名の取得
    en_name = ColorExtract.ColorMatcher.match_detailed(color_rgb, input_format="rgb")
    jp_name = ColorExtract.ColorMatcher.to_japanese(en_name)
    print(f"Color {i+1}: {en_name} ({jp_name}) - RGB: {color_rgb}")

    # パレットの描画（BGRに変換して描画）
    color_bgr = [int(c) for c in color_rgb[::-1]]
    width = int(ratio * enhanced_img.shape[1])
    cv2.rectangle(palette, (current_x, 0), (current_x + width, palette_height), color_bgr, -1)
    
    # 色名をパレット上に描画（簡易）
    cv2.putText(palette, f"{en_name}", (current_x + 5, 30), 
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
      
    current_x += width

  # 画像を縦に連結して表示（元画像、補正後、カラーパレット）
  # サイズを合わせるためにリサイズ処理が必要な場合はここで行います
  combined_view = cv2.vconcat([image, enhanced_img, palette])

  # ウィンドウ表示
  cv2.imshow("ColorExtract Test - Top: Original / Mid: CLAHE / Bottom: Palette", combined_view)
  print("\nPress any key on the image window to exit.")
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  run_test()
