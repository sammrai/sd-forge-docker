# ネット復旧ノート (ubuntu24 / Realtek 有線NIC)

最終更新: 2026-05-29

## 症状
再起動後、有線LANが繋がらない。`ping 8.8.8.8` が "Network is unreachable"、
`ip route` に default 行が無い、`ip link` に `enp4s0` が出てこない。

## 根本原因 (2026-05-29 に起きた事故)
`~/sd-forge-docker/gpu-reinstall` を実行 → `ubuntu-drivers --gpgpu install` が
NVIDIA ドライバの依存として**新カーネル(例: 6.8.0-124)を横から導入**し、起動先を
そのカーネルに切り替えた。ところがそのカーネルには NIC ドライバ入りの
`linux-modules-extra-<ver>` が付いてこず、`r8169` が無いまま起動 → ネット消失。

- ハード: Realtek RTL8111/8168 (PCI 04:00.0)、インターフェース `enp4s0`
- 必要ドライバ: `r8169` (パッケージ `linux-modules-extra-<kernel>` に含まれる)
- ゲートウェイ: 192.168.0.1

## まず状況確認 (sudo 不要)
```
uname -r                 # 今動いているカーネル
ip route | grep default  # default 行があるか
ip link                  # enp4s0 が見えるか
ls /lib/modules/$(uname -r)/kernel/drivers/net/ethernet/realtek/   # r8169.ko.zst があるか
```
`r8169.ko.zst` が無ければ「今のカーネルに NIC ドライバが無い」状態。

## 復旧A: 旧カーネルで起動してネットを取り戻す
別のカーネルに r8169 が残っているか確認:
```
for k in /lib/modules/*/; do echo "$k"; ls "$k"kernel/drivers/net/ethernet/realtek/ 2>/dev/null | grep r8169; done
```
r8169 がある旧カーネル(例: 6.8.0-117-generic)を**次回1回だけ**起動するよう指定:
```
# 正確なメニュー名は: sudo grep -E "menuentry |submenu " /boot/grub/grub.cfg
sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-117-generic"
sudo reboot
```
※ grub-reboot は次回1回限り。恒久変更ではないので安全。

## 復旧B: ネットが戻ったら壊れたカーネル用ドライバを入れる
旧カーネルで起動してネットが復活したら、問題のカーネル用に extra を入れる:
```
sudo apt update
sudo apt install linux-modules-extra-<壊れたカーネル>   # 例: 6.8.0-124-generic
# 確認:
ls /lib/modules/<壊れたカーネル>/kernel/drivers/net/ethernet/realtek/   # r8169.ko.zst が出ればOK
```
これで次回そのカーネルで起動してもネットが繋がる。

## 再発防止 (2026-05-29 に設定済み)
GPU ドライバが勝手にカーネルを巻き込まないよう hold 済み:
```
apt-mark showhold
#   nvidia-driver-580-open
#   linux-modules-nvidia-580-open-generic
```
解除する時:
```
sudo apt-mark unhold nvidia-driver-580-open linux-modules-nvidia-580-open-generic
```
※ OSカーネルの通常のセキュリティ更新は extra をセットで入れるので、そちら経由なら
  この事故は起きない。危険なのは GPU ドライバ導入がカーネルを横から引っ張る経路のみ。

## JetKVM コンソールで操作する場合の注意
物理キーボード送信だと記号(`>` `"` `:` `|`)や大文字(Shift)が化けることがある
(US/JIS 配列ずれ。例: `#`→`@`)。長いコマンドや記号が必要な時は JetKVM の
**仮想キーボード(画面上のキーをクリック)**を使うと正確に入力できる。
可能なら手元 PC から SSH してコピペするのが最も確実。
