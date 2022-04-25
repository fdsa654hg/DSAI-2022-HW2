# DSAI-2022-HW2

## 環境
python 3.6.4
Ubuntu 20.04

## 安裝套件
```
pip install -r requirements.txt
```

## 執行
```
python trader.py
```

## 資料集
·       training_data:
https://www.dropbox.com/s/uwift61i6ca9g3w/training.csv?dl=0

·       testing_data:
https://www.dropbox.com/s/duqiffdpcadu6s7/testing.csv?dl=0

## 預測模型
LSTM:適合處理時間序列問題，但股票漲跌並不是這麼單純的問題，故現實中預測準度相當低

## 方法
1. 資料標準化
2. 增加新的feature 'difference' 他的定義是'明天的開盤價'-'今天的收盤價'，會使用這個特徵的原因是要預測漲跌
3. 用過去7天的資料預測未來1天的漲跌，是二分類問題，1代表漲0則是跌
4. 這個問題相當容易overfitting 所以有使用dropout跟earlystopping
5. 設定閥值，模型輸出>=0.55取1，<=0.45則取-1，輸出介於這兩者之間或是有可能造成持股數>1或是做空次數>1的情形則取0

## 問題
1. 我第一次寫是直接預測股價(regression)，但效果不佳，跟什麼都不做的效果差不多，所以改成預測漲跌趨勢(分類)
2. train_loss雖然不難下降，但同時val_loss很容易升高，可以看出模型泛用性不高，即使在不需要付交易手續費的情形，仍然不容易獲益，可以知道在現實中股價是不太可能被預測的。
