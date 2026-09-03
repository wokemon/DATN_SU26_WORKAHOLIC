"""
Chạy script này trên máy bạn, trỏ vào file raw CSV (data/raw/lmcache_agentic_traces.csv)
Chỉ đọc thử 1 session đầu tiên để debug nhanh, không cần load cả 2.2GB.
"""
import pandas as pd

RAW_CSV_PATH = "D:\\FPT Poly\\DATN\\DATN_SU26_WORKAHOLIC\\data\\raw\\lmcache_agentic_traces.csv"  # sửa lại path cho đúng máy bạn

# Đọc thử 1 session_id đầu tiên (vài chục dòng đầu là đủ, vì file đã sort theo session)
df = pd.read_csv(RAW_CSV_PATH, nrows=60)

sid = df['session_id'].iloc[0]
g = df[df['session_id'] == sid].reset_index(drop=True)
print(f"Session test: {sid}, số turn lấy mẫu: {len(g)}")
print(f"Kiểu dữ liệu cột 'input' sau khi đọc CSV: {type(g['input'].iloc[0])}")
print()

for i in range(1, min(4, len(g))):
    cur = str(g['input'].iloc[i])
    prev = str(g['input'].iloc[i-1])
    starts_with = cur.startswith(prev)
    print(f"--- Turn {i} vs Turn {i-1} ---")
    print(f"len(prev)={len(prev)}, len(cur)={len(cur)}")
    print(f"cur.startswith(prev)? -> {starts_with}")
    if not starts_with:
        # Tìm vị trí ký tự đầu tiên khác nhau để xem lệch ở đâu
        n = min(len(prev), len(cur))
        diff_at = next((k for k in range(n) if prev[k] != cur[k]), n)
        print(f"Lệch nhau tại vị trí ký tự thứ: {diff_at}")
        print(f"prev[...]: ...{prev[max(0,diff_at-60):diff_at+60]!r}")
        print(f"cur[...]:  ...{cur[max(0,diff_at-60):diff_at+60]!r}")
    print()