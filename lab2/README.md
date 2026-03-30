# Lab 2 Binary Segmentation Project Guide

1. 這個專案在做什麼
2. 每個資料夾和檔案的用途
3. 訓練與推論的完整流程
4. 現在這版 UNet 為什麼要做 reflect padding

---

## 1. 專案在做什麼

任務是 Oxford-IIIT Pet 的二元語意分割：
- 前景: 寵物
- 背景: 非寵物

模型會輸出一張 mask，最後轉成競賽需要的 RLE 格式，存成 submission CSV。

---

## 2. 目錄總覽

- `src/`: 主要程式碼
- `src/models/`: 模型定義（UNet、ResNet34_UNet）
- `dataset/`: 訓練/驗證/測試 split 檔與資料
- `saved_models/`: 訓練好的權重與 threshold metadata
- `training_visualizations/`: 訓練時存的可視化圖與學習曲線
- `requirements.txt`: Python 相依套件

---

## 3. src 每個檔案做什麼

### 3.1 `src/train.py`

訓練進入點。

主要工作：
- 讀取 train/val dataset
- 建立模型（UNet 或 ResNet34_UNet）
- 設定 loss、optimizer、scheduler、early stopping
- 每個 epoch 做 train 與 validation
- 存 best checkpoint 和 threshold
- 存訓練可視化圖

重要設計：
- 如果是 UNet，會啟用 `pad_for_unet=True`，資料前處理先做 reflect padding。
- 如果是 ResNet34_UNet，`pad_for_unet=False`，走一般 320x320 流程。

### 3.2 `src/evaluate.py`

驗證流程與指標計算。

主要工作：
- 在 validation set 做 forward
- 算 validation loss
- 用 threshold 0.5 算 Dice
- 搜尋最佳 threshold（global dice）
- 回傳給 train.py 當選 model 的依據

重要設計：
- UNet 分支會把輸出 center crop 回 mask 空間後再算 loss/dice。
- ResNet34_UNet 分支不需要 crop。

### 3.3 `src/inference.py`

推論與產生 submission。

主要工作：
- 載入 best model 與 best threshold
- 讀測試資料
- 做 TTA（scale 與水平翻轉）
- 輸出 resize 到原圖大小
- 二值化後做 RLE，輸出 CSV

重要設計：
- UNet 在推論時會做和訓練一致的 reflect padding 邏輯。
- ResNet34_UNet 不會被 padding 分支影響。

### 3.4 `src/oxford_pet.py`

資料集讀取與前處理。

主要工作：
- 載入 image 與 trimap mask
- resize 到固定大小（預設 320x320）
- trimap 轉二元 mask（label 1 是前景）
- train 模式做 augmentation
- normalize

重要參數：
- `pad_for_unet`: 只有 UNet 會開啟
- `has_mask`: inference 時可不載入真實 mask（用 dummy mask）

### 3.5 `src/utils.py`

共用工具函式。

包含：
- seed 固定與 worker seed
- Dice 計算與 threshold 搜尋
- RLE encode
- `BCEDiceLoss`
- EarlyStopping
- 訓練曲線繪圖
- validation 預測可視化

### 3.6 `src/models/unet.py`

原版風格 UNet（無 BatchNorm、卷積不加 padding）。

特點：
- valid conv 會讓特徵圖變小
- skip connection 前需要 center crop 對齊

### 3.7 `src/models/resnet34_unet.py`

作業規格用的 ResNet34_UNet（含指定元件）。

特點：
- encoder 是 ResNet34
- decoder 有上採樣與 skip fusion
- 輸出維持 320x320

---

## 4. 訓練流程

1. 在 train.py 選模型
2. 讀取 train/val split
3. DataLoader 產生 batch
4. 模型 forward
5. 計算 BCEDiceLoss
6. backward 更新權重
7. 每個 epoch 跑 evaluate.py
8. 用最佳 validation 指標更新 best checkpoint
9. 早停條件達成則停止

---

## 5. UNet 為什麼要 reflect padding

背景：
- 原版 UNet 使用 valid conv（不加 padding）
- 如果輸入 320x320，輸出會縮小
- 直接拿縮小輸出去和 full-size mask 比，會造成評估偏差

目前方案：
- 先對 input image 做 reflect padding
- 模型照原版 UNet forward
- 輸出再 center crop 回主評估空間

這樣做的目的：
- 保持原版 UNet 的核心結構
- 同時讓訓練與評估空間一致
- 減少先前卡在低分的口徑偏差問題

---


## 6. 常用指令

### 訓練 UNet

python src/train.py --model unet

### 訓練 ResNet34_UNet

python src/train.py --model resnet34_unet

### 推論 UNet

python src/inference.py --model unet --test_txt dataset/test_unet.txt --tta_scales 1.0

### 推論 ResNet34_UNet

python src/inference.py --model resnet34_unet --test_txt dataset/test_res_unet.txt --tta_scales 1.0

---

