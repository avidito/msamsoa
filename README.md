# Modified-SAMSOA pada Multi-UAV untuk *Surveillance* dan Pemupukan Lahan Pertanian

Skripsi

Modifikasi Search-Attack Self-Organized Algorithm (SAMSOA) dari penelitian [1] untuk tugas *surveillance* dan pemupukan lahan pertanian dengan multi-UAV. UAV direpresentasikan sebagai titik dalam ruang pencarian diskrit. Pergerakan UAV dikoordinasikan dengan Ant Colony Optimization terdistribusi. Modifikasi dilakukan pada representasi target, penyederhanaan pergerakan UAV, dan penyederhanaan perhitungan. Implementasi algoritma memungkinkan pengujian kasus dengan kerusakan UAV pada saat misi berjalan. Metode dibandingkan dengan modifikasi dari metode konvensional yang terinspirasi dari penelitian [2].

## Implementasi

```
from msamsoa.algorithm.samsoa import SAMSOA_Problem
from msamsoa.utils import example as ex

scenarios = ex.scenarios
samsoa = SAMSOA_Problem(scenarios[0], na=9)
samsoa.execute()
```

## Hasil Simulasi
gif

kiri: SAMSOA, kanan: konvensional (gerak zig-zag)

dengan:
- titik sebagai UAV
- daerah putih sebagai daerah yang tidak butuh pupuk
- daerah abu-abu sebagai daerah yang membutuhkan pupuk

## Analisis Hasil

### Waktu Total Penyelesaian Tugas
Grafik

Waktu penyelesaian dari SAMSOA selalu lebih cepat dibandingkan dengan waktu penyelesaian metode konvensional (ZS). Dengan kasus terdapat kerusakan UAV, performa dari SAMSOA lebih baik dari metode konvensional ketika menggunakan lebih dari 6 UAV. Metode konvensional dengan kerusakan tidak dapat menyelesaikan misi.

### Persentase Penyelesaian pada Kasus dengan Kerusakan UAV
Grafik

Grafik merupakan hasil analisis dari pengujian dengan kerusakan UAV mencapai 30% dari seluruh UAV pada setiap simulasi. SAMSOA selalu dapat mencapai menyelesaikan tugas walaupun terdapat UAV yang rusak. Metode konvensional tidak dapat menyelesaikan tugas karena terdapat beberapa daerah yang tidak dapat diselesaikan oleh UAV rusak.

## Referensi
[1] Gao C, Zhen Z, dan Gong H. 2016. A self-organized search and attack algorithm for multiple unmanned aerial vehicles. Aerospace Science and Technology, 54, 229–240. doi:10.1016/j.ast.2016.03.022.
[2] Avellar GSC, Pereira GSA, Pimenta LCA, Iscold P. 2015. Multi-UAV Routing Area and Remote Sensing with Minimum Time. Sensors. 15(11): 27783-27803. doi:10.3390/s151127783.
