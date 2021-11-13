# IntelligentAlgorithm-CloudScheduler

#### 介绍
用智能算法解决云计算调度类问题，可以更换评价函数、绘图并存储结果和图表
需要注意的是，全都是离散调度问题，也就是说所有粒子群都是离散粒子群算法

#### 软件架构
utils.Entities里面是实验对象以及评价函数
根目录全是算法：ChaosDPSO（混沌优化离散粒子群算法）、ChaosHPSO（稍作修改的混沌优化粒子群算法）、newPSO（稍作改进提升粒子群算法）、DPSO（二进制粒子群算法，不过其实是普通的离散粒子群算法）、ACO（蚁群算法）、SA（模拟退火算法）、GA（遗传算法）、TS（禁忌搜索算法有待修改，目前可能有误）

#### 安装教程

1.  不需要安装
2.  下载下来直接在Scheduler.py使用即可


#### 使用说明

1.  实验对象参数有改动，在Entities.py里面修改，例如添加实验属性
2.  其他算法如需改动流程，在对应算法里面修改
3.  在Scheduler.py里直接修改各个算法的种群数量和迭代次数，也可统一修改
4.  是否生成数据或生成图片并保存到本地，通过在Scheduler.py注释相应代码修改
5.  粒子群的改进在于产生新解时用了一些倒置、交换、移动等等操作增强产生新解的能力

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码，注意必须统一代码风格
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
