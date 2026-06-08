# Meow 金融时序预测分析

[作业详情](https://docs.qq.com/doc/DTnhIcVFtaHhwVXVN?qqInfo=eyJtc2dJZCI6Ijc2Mzc0NDExNDc5ODczMDI1MjYiLCJtc2dUaW1lIjoiMTc3ODIzMDMzOCIsImNoYXRUeXBlIjoyLCJwZWVyVWlkIjoiMTA5MTYyMjgxMyIsInBlZXJOYW1lIjoiMjAyNuaYpS3mqKHlvI%2For4bliKvkuI7mnLrlmajlrabkuaAiLCJlbGVtSWQiOiI3NjM3NDQxMTQ3OTg3MzAyNTI1Iiwic2VuZGVyVWlkIjoidV9ZVWNBc1J4ZzhLNnkzR1h3T05nRVJ3Iiwic2VuZE5pY2tOYW1lIjoiIn0%3D&client=qqclient_online)

[github仓库](https://github.com/TreeBook-lab/Meow)

## 小组信息

组别：第7组

| 组员姓名 | 学号       | 班级序号 |
|----------|------------|----------|
| 吴天烨   | 2023302733 | 59       |
| 汪建诺   | 2023302631 | 39       |
| 张家树   | 2023302061 | 20       |

## 项目架构

```text
.
├── Images               # 项目配图、实验可视化图片
├── README.md            # 项目介绍、部署运行文档
├── environment.meow.yml # Conda环境配置文件
├── meow                 # 原始Linear Model
├── meow.py              # 项目启动入口脚本
├── meow_decoder_only    # Decoder-only Transformer Model
├── meow_lgbm            # LightGBM Model
├── meow_lstm            # LSTM Model
├── meow_self            # 自研模型
├── meow_xg              # XGBoost Model
└── requirements.txt     # pip依赖清单
```

注：当前项目中的**项目说明**、**项目报告**只是**草稿版**，最终的**项目报告**会在截止时间前提交

## 快速开始

1. 配置环境

```bash
# 1.创建虚拟环境
python -m venv .venv
# 2.激活环境
source .venv/bin/activate
# 3.再安装依赖
pip install -r requirements.txt
```

2. 运行`meow.py`

*需使用正确数据文件路径*

- 方法1：运行

```bash
python meow.py
```

可以在[meow.py](meow.py)中的 TARGET_DIR 变量中修改`"meow_decoder_only"`来设置运行的模型，默认为`Decoder-only`模型，以下是相关代码：

```python
TARGET_DIR = os.environ.get("MEOW_TARGET_DIR", "meow_decoder_only")
```

- 方法2：运行

```bash
# TARGET_DIR需替换为 meow, meow_decoder_only, meow_lgbm, meow_lstm, meow_xg中任意一个
python TARGET_DIR/meow.py
```

其中`meow/`, `meow_lstm/`, `meow_xg/`路径下`meow.py`的运行可以参考[该文件](项目说明.md)