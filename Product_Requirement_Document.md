# Product Requirements Document (PRD): Modul Deteksi Order Block (SMC)

**Dokumen:** Spesifikasi Arsitektur V5.0
**Modul:** `OrderBlockDetector`
**Sistem Inti:** AI Gold Scalper Bot (`live_trader.py`)
**Objektif:** Meminimalisir *floating loss* dengan membatasi eksekusi sinyal AI (*Action=0/1*) secara eksklusif hanya di dalam zona likuiditas tinggi (*Order Block* terkonfirmasi).

## 1. Spesifikasi Fungsional (Business Logic)

Sistem harus mampu mengidentifikasi jejak rekam pergerakan institusional (*Smart Money*) dengan mendeteksi dua anomali pergerakan harga secara berurutan:
1.  **Momentum Impulsif:** Sistem mencari *candlestick* dengan ukuran badan (*body*) yang melebihi standar volatilitas rata-rata (misalnya, > 1.5x ATR). Ini mengindikasikan adanya injeksi volume besar secara tiba-tiba.
2.  **Identifikasi Order Block (OB):** Jika momentum impulsif (poin 1) ditemukan, sistem akan menandai *candlestick* dengan arah berlawanan yang mendahului momentum impulsif tersebut. Area *High* dan *Low* dari *candlestick* berlawanan inilah yang menjadi zona OB.
    *   **Bullish OB (Demand):** *Candle bearish* terakhir sebelum ledakan *bullish*. Target untuk *BUY*.
    *   **Bearish OB (Supply):** *Candle bullish* terakhir sebelum ledakan *bearish*. Target untuk *SELL*.
3.  **Validasi FVG (Fair Value Gap):** (Opsional/Sistematis) OB dianggap valid (*High Probability*) jika pergerakan impulsif tersebut meninggalkan celah harga (FVG) yang tidak tertutup oleh sumbu (*wick*) *candlestick* sebelum dan sesudahnya.

## 2. Parameter Eksekusi (Integration Points)

*   **Penyaringan Sinyal:** Model PPO mungkin akan terus memberikan probabilitas *BUY* atau *SELL*. Sistem utama harus menolak (*override*) sinyal tersebut menjadi status `HOLD` (Abaikan) KECUALI harga saat ini (*Current Close/Bid*) berada tepat di dalam rentang harga zona OB.
*   **Timeframe Evaluasi:** Pendeteksian difokuskan pada data 20 hingga 30 *candlestick* terakhir di *timeframe* M15.

## 3. Penanganan Eksepsi (Edge Cases)
*   **Mitigation of Invalid OB:** Jika harga telah berhasil menembus dan ditutup (*candle close*) melewati batas OB (OB tertembus), zona tersebut langsung di-dinyatakan *invalid* (batal) dan dihapus dari memori untuk mencegah eksekusi yang salah arah.

---