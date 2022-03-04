
# Deep Layer Normalization
In this paper, Hongyu Wang et al. propose a simple yet effective method to stabilize extremely deep Transformers. Specifically,  Hongyu Wang et al. introduce  a  new  normalization function (DEEPNORM) to modify the residual connection in Transformer, accompanying with theoretically derived initialization. In-depth theoretical analysis shows that model updates can be bounded in a stable way. The proposed method combines the best of two worlds, i.e., good performance of Post-LN and stable training of Pre-LN, making DEEPNORM a preferred alternative. We successfully scale Transformers upto 1,000 layers (i.e., 2,500 attention and feed-forward network sublayers) without difficulty, which is one order of magnitude deeper than previous deep Transformers. Remarkably, on a multilingual benchmark with 7,482 translation directions, their 200-layer model with 3.2B parameters significantly outperforms the 48-layer state-of-the-art model with 12B parameters by 5 BLEU points, which indicates a promising scaling direction.

# 深度层归一化
本文提出了一种简单而有效的方法来稳定极深变换器。具体来说，作者引入了一个新的归一化函数(DEEPNORM)修改变换器中的残差连接，伴随理论上推导出的初始化。深入的理论分析表明，模型更新可以以稳定的方式进行限制。所提出的方法结合了两全其美，即Post-LN的良好表现和Pre-LN的稳定训练，使DEEPNORM成为首选替代方案。作者成功地扩大了变换器到1,000层（即 2,500 个注意和前馈网络子层）难度，比之前的深度变换器要深一个数量级。值得注意的是，在包含7,482个翻译方向的多语言基准测试中，作者具有3.2B参数的200层模型，显着优于具有120亿个参数当前最佳48层模型，优势达5个BLEU点，这表明深度是有希望的扩展方向。

# Reference
[1] https://arxiv.org/pdf/2203.00555.pdf  
[2] https://hub.baai.ac.cn/view/15195