## 车牌分割（传统算法）

RGB图像->边界扩充->灰度变化->阈值处理->   
轮廓检测->边缘检测->Haugh直线检测->倾斜校正->   
水平投影->分割线->车牌分割

## Requirements

Ubuntu 18.04  
Inter(R) Core(TM) i7-8700 CPU @ 3.2GHz  
Python: 3.7  
Opencv: 4.0.0

## Results
样本         数量      Fps         Accuracy     
挂号车牌     2000      188.57      94.7%

## 存在的问题
1. 无边缘、 有阴影的车牌分割效果不好;
2. 双行车牌中部分含 Q 字符的分割效果不好， 右下角的点不清楚， 车牌字符中没有 O 字符;
3. 建议提供包含车牌边缘的数据;

## Update Version 
2019-06-04
