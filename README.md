## DLLM推理加速

最终目标是实现一种快速attention算法，在多个DLLM上拿到推理加速。先在LaViDa-L模型上试通。

## 这套代码用法

用lmms-eval库测评共12个数据集，others_L_2.sh指 DLLM的 max_new_token是 2，其他同理。

```bash
cd LaViDa-main
bash eval/others_L_2.sh
```

可视化attn map，可以可视化每个 decode step下的 attn map，以及每个模型 layer在不同 decode step下的attn map数值差距折线图。

```bash
cd LaViDa-main
python plot_semi.py
```