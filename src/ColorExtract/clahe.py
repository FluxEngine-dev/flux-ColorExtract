import numpy as np
import cv2
from cv2.typing import MatLike, Size

class Clahe:
  """
  CLAHE (Contrast Limited Adaptive Histogram Equalization) による
  局所コントラスト補正を簡易に行うクラス。

  Args:
    clip_limit (float, optional): コントラスト制限の強さ。値を大きくすると強調が強くなる。デフォルトは 2.0。
    tile_grid_size (Size, optional): 画像をブロック分割するサイズ。小さいほど細かく調整される。デフォルトは (8, 8)。

  Examples:
    >>> import cv2
    >>> from color_extract.clahe import Clahe
    >>> img = cv2.imread("sample.jpg")
    >>> clahe = Clahe(clip_limit=2.5, tile_grid_size=(8, 8))
    >>> enhanced = clahe.apply_yuv(img)
    >>> cv2.imwrite("output.jpg", enhanced)
  """
  def __init__(self
    , clip_limit: float=2.0        # コントラスト制限の強さ（値を大きくすると強調が強くなる）
    , tile_grid_size: Size=(8,8)  # 画像を分割して局所的に処理するブロックサイズ（小さいほど細かく調整）
  ):
    # クラス内変数 ===================
    self._clahe:cv2.CLAHE = None
    # ===============================
    self._clahe = cv2.createCLAHE(
        clipLimit=clip_limit
      , tileGridSize=tile_grid_size)

  def apply_yuv(self
    , img_bgr: MatLike
  ) -> MatLike:
    """
    YUV色空間の輝度成分（Y）に対してCLAHEを適用。

    Args:
      img_bgr (MatLike): BGR画像。

    Returns:
      MatLike: CLAHE適用後のBGR画像。
    """
    if img_bgr is None:
      return None
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = self._clahe.apply(img_yuv[:,:,0])  # Y成分
    result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return result

  def apply_hsv(self
    , img_bgr: MatLike
    , equalize_v: bool=True
    , equalize_s: bool=False
  ) -> MatLike:
    """
    HSV色空間に変換し、彩度(S)や明度(V)にCLAHEを適用。

    Args:
      img_bgr (MatLike): BGR画像。
      equalize_v (bool): 明度チャンネルに適用するか。デフォルト True。
      equalize_s (bool): 彩度チャンネルに適用するか。デフォルト False。

    Returns:
      MatLike: CLAHE適用後のBGR画像。
    """
    if img_bgr is None:
      return
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if equalize_s:
      img_hsv[:,:,1] = self._clahe.apply(img_hsv[:,:,1])
    if equalize_v:
      img_hsv[:,:,2] = self._clahe.apply(img_hsv[:,:,2])
    result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return result
