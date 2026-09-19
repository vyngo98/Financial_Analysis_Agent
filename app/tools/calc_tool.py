"""Công cụ tính % tất định (Python) cho câu hỏi tỷ suất / tăng trưởng.

qwen2:7b chia số nhiều-chữ-số không đáng tin (câu "biên lợi nhuận gộp", "tăng trưởng %").
Agent trích ĐÚNG hai con số từ kết quả search rồi gọi tool này để tính CHÍNH XÁC, thay vì
tự nhẩm. Deterministic, không phụ thuộc LLM.
"""

from langchain.tools import tool


def _fmt(pct: float) -> str:
    """Làm tròn 1 chữ số thập phân, dấu phẩy thập phân kiểu VN (35.9 -> '35,9')."""
    return f"{pct:.1f}".replace(".", ",")


@tool
def calculate_percentage(
    numerator: float,
    denominator: float,
    mode: str = "ratio",
) -> str:
    """Tính phần trăm CHÍNH XÁC từ hai con số (dùng cho câu hỏi tỷ suất / tăng trưởng).

    Dùng công cụ này BẤT CỨ KHI NÀO câu hỏi cần một tỷ lệ phần trăm (biên lợi nhuận,
    tỷ trọng, mức tăng/giảm %) — đừng tự tính nhẩm phép chia.

    Args:
        numerator: Con số tử (ratio) HOẶC giá trị KỲ NÀY / năm nay (growth).
        denominator: Con số mẫu (ratio) HOẶC giá trị KỲ TRƯỚC / năm trước (growth).
        mode: "ratio" = numerator/denominator*100 (vd biên LN gộp, tỷ trọng);
              "growth" = (numerator-denominator)/denominator*100 (mức tăng/giảm so kỳ trước).
    """
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return "Không tính được: hai giá trị phải là số."
    if den == 0:
        return "Không tính được: mẫu số bằng 0."

    m = (mode or "ratio").strip().lower()
    if m == "growth":
        pct = (num - den) / den * 100.0
        chieu = "tăng" if pct >= 0 else "giảm"
        return f"≈ {_fmt(abs(pct))}% ({chieu} so với kỳ trước)"
    # mặc định ratio
    pct = num / den * 100.0
    return f"≈ {_fmt(pct)}%"
