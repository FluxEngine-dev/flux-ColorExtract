import numpy as np
import cv2
from cv2.typing import MatLike, Size
from typing import Sequence
from sklearn.cluster import KMeans

class ColorFeatureExtractor_Histogram:
  """
  BGR画像から各チャンネルのヒストグラムを計算するクラス。

  Args:
    image (MatLike): 入力BGR画像。
    contours (Sequence[MatLike], optional): 特定領域のみを解析したい場合の輪郭。

  Examples:
    >>> extractor = ColorFeatureExtractor_Histogram(image)
    >>> hist = extractor.get_histograms()
    >>> print(hist["r"].shape)
    (256,)
  """
  def __init__(self
    , image: MatLike
    , contours: Sequence[MatLike] = None
  ):
    self._histograms = {}
    if image is None:
      return

    # --- 輪郭マスク生成 ---
    if contours is not None and len(contours) > 0:
      mask = np.zeros(image.shape[:2], dtype=np.uint8)
      cv2.drawContours(mask, contours, -1, color=255, thickness=-1)
      pixels = image[mask == 255]
    else:
      pixels = image.reshape((-1, 3))
    
    if len(pixels) == 0:
      return
    
    pixels = pixels.reshape((-1,1,3))

    color = ("b", "g", "r")
    for i, col in enumerate(color):
      # ヒストグラムを計算 (特徴量)
      hist = cv2.calcHist([pixels], [i], None, [256], [0, 256])
      self._histograms[col] = hist.flatten()

  def get_histograms(self) -> dict:
    """B, G, R 各チャンネルのヒストグラムを返す。"""
    return self._histograms

class ColorFeatureExtractor_KMeans:
  """
  BGR画像からK-Means法により代表色と割合を抽出するクラス。

  Args:
    image (MatLike): 入力BGR画像。
    contours (Sequence[MatLike], optional): 特定領域の輪郭。
    n_clusters (int, optional): クラスタ数。デフォルトは3。
    random_state (int, optional): 乱数シード。デフォルトは42。

  Examples:
    >>> extractor = ColorFeatureExtractor_KMeans(image, n_clusters=3)
    >>> colors = extractor.get_dominant_colors()
    >>> ratios = extractor.get_ratios()
  """
  def __init__(self
    , image: MatLike
    , contours: Sequence[MatLike] = None
    , n_clusters: int=3
    , random_state: int=42
  ):
    self._dominant_colors = None
    self._ratios = None

    if image is None:
      return

    # --- 輪郭マスク生成 ---
    if contours is not None and len(contours) > 0:
      mask = np.zeros(image.shape[:2], dtype=np.uint8)
      cv2.drawContours(mask, contours, -1, color=255, thickness=-1)
      pixels = image[mask == 255]
    else:
      pixels = image.reshape((-1, 3))

    # BGRからRGBに変換し、ピクセル配列を作成
    pixels = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_BGR2RGB).reshape(-1,3)

    if len(pixels) == 0:
      return

    # K-means
    kmeans = KMeans(
        n_clusters=n_clusters  # 抽出する色数
      , random_state=random_state # 乱数シード固定
      , n_init="auto"
    ).fit(pixels)
    
    # 代表色 (RGB)
    self._dominant_colors = np.array(kmeans.cluster_centers_, dtype="uint8")
    labels = kmeans.labels_

    # 各色の割合計算
    _, counts = np.unique(labels, return_counts=True)
    self._ratios = counts / counts.sum()

  def get_dominant_colors(self) -> np.ndarray:
    """代表色 (RGB) のリストを返す。"""
    return self._dominant_colors

  def get_ratios(self) -> np.ndarray:
    """各代表色の割合を返す。"""
    return self._ratios

  @classmethod
  def warmup(cls):
    """初回のKMeans実行をキャッシュするための軽量ウォームアップ。"""
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    pixels = img.reshape(-1, 3)
    _ = KMeans(n_clusters=1, random_state=42, n_init="auto").fit(pixels)

import numpy as np
import cv2
from cv2.typing import MatLike, Size
from typing import Sequence
from sklearn.cluster import KMeans

class ColorFeatureExtractor_Histogram:
  """
  BGR画像から各チャンネルのヒストグラムを計算するクラス。

  Args:
    image (MatLike): 入力BGR画像。
    contours (Sequence[MatLike], optional): 特定領域のみを解析したい場合の輪郭。

  Examples:
    >>> extractor = ColorFeatureExtractor_Histogram(image)
    >>> hist = extractor.get_histograms()
    >>> print(hist["r"].shape)
    (256,)
  """
  def __init__(self
    , image: MatLike
    , contours: Sequence[MatLike] = None
  ):
    self._histograms = {}
    if image is None:
      return

    # --- 輪郭マスク生成 ---
    if contours is not None and len(contours) > 0:
      mask = np.zeros(image.shape[:2], dtype=np.uint8)
      cv2.drawContours(mask, contours, -1, color=255, thickness=-1)
      pixels = image[mask == 255]
    else:
      pixels = image.reshape((-1, 3))
    
    if len(pixels) == 0:
      return
    
    pixels = pixels.reshape((-1,1,3))

    color = ("b", "g", "r")
    for i, col in enumerate(color):
      # ヒストグラムを計算 (特徴量)
      hist = cv2.calcHist([pixels], [i], None, [256], [0, 256])
      self._histograms[col] = hist.flatten()

  def get_histograms(self) -> dict:
    """B, G, R 各チャンネルのヒストグラムを返す。"""
    return self._histograms

class ColorFeatureExtractor_KMeans:
  """
  BGR画像からK-Means法により代表色と割合を抽出するクラス。

  Args:
    image (MatLike): 入力BGR画像。
    contours (Sequence[MatLike], optional): 特定領域の輪郭。
    n_clusters (int, optional): クラスタ数。デフォルトは3。
    random_state (int, optional): 乱数シード。デフォルトは42。

  Examples:
    >>> extractor = ColorFeatureExtractor_KMeans(image, n_clusters=3)
    >>> colors = extractor.get_dominant_colors()
    >>> ratios = extractor.get_ratios()
  """
  def __init__(self
    , image: MatLike
    , contours: Sequence[MatLike] = None
    , n_clusters: int=3
    , random_state: int=42
  ):
    self._dominant_colors = None
    self._ratios = None

    if image is None:
      return

    # --- 輪郭マスク生成 ---
    if contours is not None and len(contours) > 0:
      mask = np.zeros(image.shape[:2], dtype=np.uint8)
      cv2.drawContours(mask, contours, -1, color=255, thickness=-1)
      pixels = image[mask == 255]
    else:
      pixels = image.reshape((-1, 3))

    # BGRからRGBに変換し、ピクセル配列を作成
    pixels = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_BGR2RGB).reshape(-1,3)

    if len(pixels) == 0:
      return

    # K-means
    kmeans = KMeans(
        n_clusters=n_clusters  # 抽出する色数
      , random_state=random_state # 乱数シード固定
      , n_init="auto"
    ).fit(pixels)
    
    # 代表色 (RGB)
    self._dominant_colors = np.array(kmeans.cluster_centers_, dtype="uint8")
    labels = kmeans.labels_

    # 各色の割合計算
    _, counts = np.unique(labels, return_counts=True)
    self._ratios = counts / counts.sum()

  def get_dominant_colors(self) -> np.ndarray:
    """代表色 (RGB) のリストを返す。"""
    return self._dominant_colors

  def get_ratios(self) -> np.ndarray:
    """各代表色の割合を返す。"""
    return self._ratios

  @classmethod
  def warmup(cls):
    """初回のKMeans実行をキャッシュするための軽量ウォームアップ。"""
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    pixels = img.reshape(-1, 3)
    _ = KMeans(n_clusters=1, random_state=42, n_init="auto").fit(pixels)
