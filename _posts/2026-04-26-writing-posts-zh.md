---
title: "写文章的语法速查 —— 数学、代码、提示框、图片"
date: 2026-04-26 09:30:00 +0800
categories: [杂记, 教程]
tags: [jekyll, chirpy, markdown, latex]
math: true
mermaid: true
lang: zh-CN
permalink: /posts/writing-posts/
---

写给我自己（也给后来者）的速查 —— 这个 Chirpy 主题的博客支持哪些 Markdown 语法。文章顶部的 front-matter 是这样：

```yaml
---
title: "你的标题"
date: 2026-04-26 12:00:00 +0800
categories: [一级, 二级]
tags: [标签1, 标签2]
math: true        # 文章里有 LaTeX 时打开
mermaid: true     # 文章里有 mermaid 图时打开
pin: true         # 置顶到首页
lang: zh-CN       # 语言：en 或 zh-CN
permalink: /posts/your-slug/   # 可选，固定 URL
image:
  path: /assets/img/cover.png
  alt: 封面
---
```

## 数学公式（LaTeX）

行内：$\mathcal{L}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$。

行间：

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, A^{\pi}(s_t, a_t)\right]
$$

经典的策略梯度恒等式 —— REINFORCE、A2C、PPO 以及大部分机器人 RL 都建立在它之上。

## 带行号的代码块

```python
import torch
import torch.nn as nn

class TinyPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))
```

Shell 片段也没问题：

```bash
# 启动训练
python train.py --config configs/ppo_humanoid.yaml --seed 42
```

## 提示框（Callouts）

> 信息提示 —— 链接、参考、文档指引。
{: .prompt-info }

> 经验之谈 —— 别忘了给 dataloader 设 seed。
{: .prompt-tip }

> 警告 —— 这个操作在 batch size 上是 O(N²)，要么把 N 限住要么会 OOM。
{: .prompt-warning }

> 危险 —— 别在生产集群上不做演练就直接跑。
{: .prompt-danger }

## Mermaid 流程图

```mermaid
flowchart LR
    A[原始演示数据] --> B[Tokenize]
    B --> C[预训练 VLA]
    C --> D{评测通过?}
    D -- 是 --> E[RL 微调]
    D -- 否 --> B
    E --> F[部署到机器人]
```

## 图片

```markdown
![alt text](/assets/img/example.png){: width="600" }
_图片说明放这里。_
```

## 表格

| 方法 | 实时? | 样本效率 | 训练时长 |
|------|:----:|:-------:|:-------:|
| BC   | ✅   | 高      | 低       |
| PPO  | ❌   | 低      | 高       |
| Diffusion Policy | ✅ | 中  | 中       |

## 脚注

回放缓冲区[^buffer] 用来存储 off-policy 学习需要的转移。

[^buffer]: 一个 FIFO 数据结构，存的是 `(s, a, r, s', done)` 四元组，连续控制场景下通常开到 1M 大小。

---

工具齐了，去写吧。
