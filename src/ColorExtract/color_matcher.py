import matplotlib.colors as mcolors
import numpy as np
from typing import Literal, Tuple

from ColorExtract.color_matcher_jp_map import (
  COLOR_NAME_JP_MAP,
  COLOR_BASE_NAME_JP_MAP
)

class ColorMatcher:
  """
  RGB または BGR 値から、最も近い色名を検索するクラス。

  CSS4 / BASE_COLOR に基づく距離マッチングを行い、
  英語名または日本語名を返す。

  Examples:
    >>> from color_extract.color_matcher import ColorMatcher
    >>> name_en = ColorMatcher.match_detailed((255, 0, 0), input_format="rgb")
    >>> name_jp = ColorMatcher.to_japanese(name_en)
    >>> print(name_en, name_jp)  # red, 赤
  """
  CSS4_BGR_COLORS = {}
  BASE_BGR_COLORS = {}
  def __colors_set(mcolor_colors: dict, set: dict):
    """matplotlibカラー辞書からBGR形式の辞書を生成。"""
    for name, hex in mcolor_colors.items():
      f_rgb = mcolors.hex2color(hex)
      set[name] = [
          int(f_rgb[2] * 255)
        , int(f_rgb[1] * 255)
        , int(f_rgb[0] * 255) ]
  __colors_set(mcolors.CSS4_COLORS, CSS4_BGR_COLORS)
  __colors_set(mcolors.BASE_COLORS, BASE_BGR_COLORS)

  @classmethod
  def to_japanese(cls, english_name: str):
    """英語の色名を日本語に変換する。"""
    name_lower = english_name.lower()
    jp_name = COLOR_NAME_JP_MAP.get(name_lower)
    if jp_name:
      return jp_name
    jp_name = COLOR_BASE_NAME_JP_MAP.get(name_lower)
    if jp_name:
      return jp_name
    return english_name

  @classmethod
  def _match_core(cls
    , color_dict: dict
    , color: Tuple[int, int, int]
    , input_format
  ) -> str:
    """指定色空間で最も近い色名を検索する内部関数。"""
    c1, c2, c3 = int(color[0]), int(color[1]), int(color[2])
    
    if input_format.lower() == "rgb":
      i_r, i_g, i_b = c1, c2, c3
    elif input_format.lower() == "bgr":
      i_b, i_g, i_r = c1, c2, c3
    else:
      raise ValueError("input_format は 'bgr' または 'rgb' で指定してください。")
        
    min_distance_sq = float("inf")
    closest_color_name = "Unknown"
    
    for name, bgr_known in color_dict.items():
      k_b, k_g, k_r = int(bgr_known[0]), int(bgr_known[1]), int(bgr_known[2])
      
      # ユークリッド距離の二乗を計算 (平方根の計算は重いため、比較時は二乗のまま)
      distance_sq = (i_b - k_b)**2 + (i_g - k_g)**2 + (i_r - k_r)**2
      
      if distance_sq < min_distance_sq:
        min_distance_sq = distance_sq
        closest_color_name = name
    
    return closest_color_name

  @classmethod
  def match_detailed(cls
    , color: Tuple[int,int,int]
    , input_format: Literal["bgr", "rgb"] = "bgr"
  ) -> str:
    """CSS4の全色から最も近い色名を検索。"""
    return cls._match_core(cls.CSS4_BGR_COLORS, color, input_format)
  
  @classmethod
  def match_base(cls
    , color
    , input_format: Literal["bgr", "rgb"] = "bgr"
  ):
    """基本色（少数の主要色）から最も近い色名を検索。"""
    return cls._match_core(cls.BASE_BGR_COLORS, color, input_format)
