---
title: 基于DNN的艺术风格生成算法的应用
date: 2016-05-06 22:00:00
categories: 
	- Research
	- DeepLearning
	- APP
tags: [DeepLearning,APP,Art]
#copyright: false
---
## 富有艺术感的Paper
2015年，德国科学家发表了一篇名为[《A Neural Algorithm of Artistic Style》](http://arxiv.org/abs/1508.06576)的论文，它使用深度学习算法，将普通图片进行加工，便可以创造出指定图片风格的作品。
<!-- more -->
艺术家可能需要用毕生的心血才能创造出惊人的艺术作品，而论文所述算法便可以用很短的时间将普通图片创造出富有大师风格的画作。


## 实验室产品
据此论文，我们实验室研发了一个应用，并且和其它公布出来的源码做了对比。

### Style
本人并不是很喜欢欧美抽象派的画作，倒是对国内大师的中国风作品颇有好感，于是我选了吴冠中《小桥流水人家》系列的一幅作品做为*style*，虽远不及最近拍卖的《周庄》出名，但是整体感觉要好，颜色也更丰富了些，展现江南水乡的美。
{% asset_img style_.jpg 吴冠中《小桥流水人家》 %}

### Content
*content*图片的内容一般来说要和style的内容接近才好，当然随意的搭配也可能产生神奇的效果。本人在广州读的本硕，自然想到了岭南建筑，这里选了广州市大学城岭南印象园的一张图片。好吧，我承认其实这张照片不太好看。但是，如果最后跑出来的合成结果效果不错的话，不就更能说明算法和应用的功效吗？
{% asset_img content_.jpg 广州市大学城岭南印象园 %}

### Result
经过150次的迭代，结果如下图。原图搭配吴冠中的画风，普通照相技术下的照片被“美化”了，有木有？！这个‘学习’出来的效果真的不错，原本青砖纹路的墙壁被渐渐刷白，留下屋檐的轮廓。街上的人物、灯笼等，也和原图一样，是一个个的彩色小点。
{% asset_img painting_.jpg “吴冠中的”《岭南印象园》 %}


## 其他源码
笔者暂时装好了下面两个:
### neural-style
 [`https://github.com/jcjohnson/neural-style`](https://github.com/jcjohnson/neural-style "neural-style")
neural-style产生的效果差不多，它对原图的色彩保留较好，测试1000像素图片迭代500次大概7分钟+；但是我们代码的黑色墨笔勾勒地更深，更接近于吴冠中的风格。
**安装：**
按照教程[`https://github.com/jcjohnson/neural-style/blob/master/INSTALL.md`](https://github.com/jcjohnson/neural-style/blob/master/INSTALL.md "https://github.com/jcjohnson/neural-style/blob/master/INSTALL.md")
出现问题：`安装loadcaffe时，出现Error: Failed cloning git repositor`
解决办法：`执行:"git config --global url."https://".insteadOf git://"`
{% asset_img neural-style_500_.jpg neural-style迭代500次效果 %}

### neuralart
 [`https://github.com/kaishengtai/neuralart`](https://github.com/kaishengtai/neuralart "neuralart")
neuralart的效果就没那么好，且消耗内存巨大（因为同样的设置居然跑不动，非得设置成sgd），但是隐隐约约我能感受到它很适用于水粉画和印象派。
**安装：**
安装教程：[`https://github.com/kaishengtai/neuralart/blob/master/README.md`](https://github.com/kaishengtai/neuralart/blob/master/README.md "https://github.com/kaishengtai/neuralart/blob/master/README.md")
但它需要用到 *qlua*， *qlua* 默认没有安装，且依赖 *lua5.2*，所以将 *neural-style* 安装时在 *torch* 下默认安装的 *lua5.1* 换成 *lua5.2*，根据[`http://torch.ch/docs/getting-started.html`](http://torch.ch/docs/getting-started.html)。
另外，源代码默认是启用GPU 0，所以修改源代码添加对GPU的设置。
{% asset_img neuralart_500_.jpg neuralart迭代500次效果 %}