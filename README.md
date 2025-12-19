# ColorExtract

画像コントラスト強調（CLAHE）、K-Meansによる支配色抽出、多言語カラーネームマッチングのためのPythonモジュール

## 特徴

* **局所コントラスト補正 (CLAHE):** YUV/HSV色空間を利用し、自然な明るさとコントラストの調整が可能。
* **高度な色抽出:** K-Meansクラスタリングを用いて、画像から主要な色（ドミナントカラー）とその割合を抽出。
* **多言語色名マッチング:** 抽出したRGB/BGR値に最も近い色名を推定。英語および日本語に対応。
* **特定領域解析:** 輪郭（Contours）を指定することで、画像内の特定の部分だけを解析対象にすることが可能。

## インストール

### GitHubから直接

```bash
pip install git+https://github.com/FluxEngine-dev/flux-ColorExtract.git
```

### 開発用（リポジトリをクローンした後）

```bash
pip install -e .
```

## 使い方

```python
import cv2
from ColorExtract import Clahe, ColorFeatureExtractor_KMeans, ColorMatcher

# 1. 画像の読み込みとコントラスト補正
img = cv2.imread("sample.jpg")
clahe = Clahe(clip_limit=2.0, tile_grid_size=(8, 8))
enhanced_img = clahe.apply_yuv(img)

# 2. 代表色の抽出 (K-Means)
extractor = ColorFeatureExtractor_KMeans(enhanced_img, n_clusters=3)
dominant_colors = extractor.get_dominant_colors()  # RGB値のリスト
ratios = extractor.get_ratios()                  # 各色の占有割合

# 3. 色名のマッチング (英語・日本語)
for color in dominant_colors:
    name_en = ColorMatcher.match_detailed(color, input_format='rgb')
    name_jp = ColorMatcher.to_japanese(name_en)
    print(f"Color: {color} -> {name_en} ({name_jp})")
```

## 主要API一覧

| カテゴリ | クラス/メソッド | 説明 |
| --- | --- | --- |
| **補正** | `Clahe.apply_yuv(img)` | YUV空間のY成分に対してCLAHEを適用し、自然な補正を行う。 |
| **補正** | `Clahe.apply_hsv(img)` | HSV空間のV成分に対してCLAHEを適用する。 |
| **抽出** | `ColorFeatureExtractor_KMeans` | K-Means法を用いて主要な色と占有割合を計算する。 |
| **抽出** | `ColorFeatureExtractor_Histogram` | 各チャンネルの輝度ヒストグラムを取得する。 |
| **推定** | `ColorMatcher.match_detailed` | RGB/BGR値からCSS4カラーネームを検索する。 |
| **推定** | `ColorMatcher.to_japanese` | 英語の色名を対応する日本語名に変換する。 |

## 依存関係

* Python 3.10+
* OpenCV (`opencv-python`)
* NumPy
* scikit-learn
* Matplotlib (カラーマップ処理用)

## 制限事項

* `cv2.putText` を使用した日本語表示には対応していません（コンソール出力やPillow経由の描画を推奨）。
* 複雑な背景を持つ画像では、K-Meansのクラスタ数（`n_clusters`）の調整が必要になる場合があります。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細については、[LICENSE](LICENSE)ファイルをご覧ください。

Copyright (c) 2025 [FluxEngine-dev]
