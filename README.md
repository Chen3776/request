## DLLM推理加速

目标是实现快速attention算法，在多个DLLM上拿到推理加速。先在LaViDa-L模型上试通。

## 这套代码用法

用lmms-eval库测评共12个数据集，others_L_2.sh指 DLLM的 max_new_token是 256，其他同理。

```bash
cd LaViDa-main
bash eval/others_L_256.sh
```

可视化attn map，可以可视化每个 decode step下的 attn map，以及每个模型 layer在不同 decode step下的attn map数值差距折线图。

```bash
cd LaViDa-main
python plot_semi.py
```

测速：最简单的测速脚本，是调用自定义attention函数并计时。

```bash
cd LaViDa-main
python time3.py
```
