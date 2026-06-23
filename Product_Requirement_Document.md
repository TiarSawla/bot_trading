**Project:** AI Gold Bot V3.0  
**Module:** `src/live_trader.py`  
**Objective:** Mengubah `volume` order yang di-_hardcode_ (0.01) menjadi volume dinamis yang dihitung berdasarkan _Free Margin_ akun dengan metode _Equity Step_, dibatasi oleh _Max Lot_ demi keamanan komputasi.

## 1. Spesifikasi Matematis:

- **Equity Step:** `$1000`
- **Base Lot:** `0.01`
- **Max Lot (Safety Clamp):** `0.10`
- **Formula:** `calculated_lot = math.floor(margin_free / 1000) * 0.01`
- Lot akhir harus divalidasi agar tidak kurang dari `0.01` dan tidak lebih dari `0.10`, lalu dibulatkan 2 desimal (`round(lot, 2)`).

---
