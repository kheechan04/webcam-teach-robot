"""조건 ⑤용 인쇄 마커(ArUco) 만들기 — A4 PDF.

    uv run python scripts/make_marker.py
→ docs/marker/wrist_marker_A4.pdf (인쇄용), docs/marker/wrist_marker_50mm.png

마커: OpenCV ArUco DICT_4X4_50의 0번, 검은 사각형 한 변 50 mm(MARKER_MM). 둘레에 흰 여백이 있어야 인식된다.
**반드시 "실제 크기(100%)"로 인쇄**하고, 종이의 100 mm 눈금선을 자로 재서 확인한다(크기가 틀리면 깊이가 그 비율로 틀린다).
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "marker"
DPI = 300
MARKER_MM = 50.0
MARKER_ID = 0
A4_MM = (210, 297)


def mm(v: float) -> int:
    return int(round(v / 25.4 * DPI))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = mm(MARKER_MM)
    marker = cv2.aruco.generateImageMarker(dic, MARKER_ID, side)
    Image.fromarray(marker).save(OUT / "wrist_marker_50mm.png", dpi=(DPI, DPI))

    page = Image.new("L", (mm(A4_MM[0]), mm(A4_MM[1])), 255)
    d = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("malgun.ttf", mm(4))
        small = ImageFont.truetype("malgun.ttf", mm(3))
    except OSError:
        font = small = ImageFont.load_default()
    d.text((mm(15), mm(12)), "손목 마커 (ArUco 4x4_50, id 0, 검은 사각형 한 변 50 mm)", fill=0, font=font)
    d.text((mm(15), mm(20)), "실제 크기(100%)로 인쇄. 아래 눈금선이 정확히 100 mm인지 자로 확인.", fill=0, font=small)
    # 100 mm 확인용 눈금선
    y = mm(32)
    d.line([(mm(15), y), (mm(115), y)], fill=0, width=mm(0.4))
    for k in range(0, 101, 10):
        d.line([(mm(15 + k), y - mm(2 if k % 50 else 4)), (mm(15 + k), y)], fill=0, width=mm(0.3))
    d.text((mm(118), y - mm(3)), "100 mm", fill=0, font=small)
    # 마커 두 장 (하나는 예비). 흰 여백 10 mm + 자를 선
    for i, top in enumerate((45, 140)):
        x0, y0 = mm(15), mm(top)
        pad = mm(10)
        d.rectangle([x0, y0, x0 + side + 2 * pad, y0 + side + 2 * pad], outline=0, width=2)
        page.paste(Image.fromarray(marker), (x0 + pad, y0 + pad))
        d.text((x0 + side + 2 * pad + mm(5), y0 + mm(5)), f"{'예비' if i else '사용'}: 테두리 선을 따라 자르기", fill=0, font=small)
        d.text((x0 + side + 2 * pad + mm(5), y0 + mm(12)), "흰 여백은 남겨 두세요", fill=0, font=small)
    d.text((mm(15), mm(240)), "붙이는 곳: 손목 안쪽(손바닥 쪽), 카메라를 향하게. 평평하게, 구겨지지 않게.", fill=0, font=small)
    d.text((mm(15), mm(247)), "가능하면 두꺼운 종이·카드에 붙이고 손목에 테이프나 고무줄로 고정.", fill=0, font=small)
    page.save(OUT / "wrist_marker_A4.pdf", resolution=DPI)
    print(f"→ {OUT / 'wrist_marker_A4.pdf'}  (마커 {side} px = {MARKER_MM} mm @ {DPI} dpi)")


if __name__ == "__main__":
    main()
