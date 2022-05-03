本文件夹主要程序为
dtrg.py（简单可微分张量网络算法） 和 
dTRGB.py（考虑末次幻境的可微分张良网络算法）
其都是应用于2DIsing Model计算最后的自由能。

eig.py  : 利用power method计算出矩阵的最大本征值
expm.py : 计算e^A, 其中A为矩阵
rig.py : 2DIsing Model的一些温度点下的onsager严格解
TorNcon.py : 计算张量收缩的工具库
AsymLogm.py : 由酉矩阵U（规范）得到反对称矩阵A，e^A = U

直接运行dtrg or dtrgB应该就能跑通，并可以在这两个文件后面
修改主程序参数，优化次数，重整化次数，温度点等参数。

如果程序跑不通或者有别的问题，欢迎邮件联系
derricfan@gmail.com 

