# 医疗体检数据健康风险评估系统

一个基于 Flask + 逻辑回归 + 千问大模型的全栈健康风险评估系统，实现用户健康数据录入、高血压/糖尿病风险预测、可视化图表展示与 AI 个性化健康建议。

## 📌 项目简介

本项目是一个全栈式医疗健康风险评估平台，用户注册登录后，可自主录入年龄、血压、血糖、BMI、糖化血红蛋白等多项体检指标。系统调用预训练的逻辑回归模型，自动选择专业或通用模型进行高血压与糖尿病风险评估，输出风险等级与置信度，并通过 Plotly 图表直观展示。同时接入阿里云千问大模型，根据用户数据与预测结果生成个性化健康建议报告，帮助用户快速了解自身健康风险并获取改善建议。

## 🎯 核心功能

| 功能模块 | 说明 |
|---------|------|
| 👤 用户管理 | 注册、登录、会话管理，密码哈希存储 |
| 📝 健康数据录入 | 支持 14 项健康指标，实时输入校验，性别联动怀孕次数，快捷场景预设 |
| 🧠 智能风险评估 | 双模型策略（糖尿病专业/通用），自动选择，缺失值用人群平均值填充 |
| 📊 结果可视化 | Plotly 置信度对比图，风险等级徽章，置信度进度条 |
| 🤖 AI 健康顾问 | 千问大模型接入，自动生成风险解读、生活建议、就医提醒、鼓励话语 |
| 📄 报告导出 | 一键将评估报告导出为 PNG 图片 |
| 🐳 Docker 部署 | 支持 `docker-compose up -d` 一键启动，包含 Flask 应用与 MySQL 数据库 |
| 📱 响应式设计 | 支持 PC 端和移动端访问 |

## 🛠️ 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 后端框架 | Flask | 2.3.0 |
| 数据库 | MySQL | 8.0 |
| 机器学习 | scikit-learn（逻辑回归） | 1.2.2 |
| 数据处理 | pandas + numpy | 1.5.3 + 1.24.3 |
| 模型持久化 | joblib | 1.2.0 |
| 大模型 | 千问 API（qwen-turbo） | - |
| 前端 | HTML5/CSS3/JavaScript + Plotly + html2canvas | - |
| 容器化 | Docker + docker-compose | - |

## 📁 项目结构

```
PythonProject（doctor）/
├── data/                          # 数据集
│   ├── 高血压数据集.csv
│   ├── 糖尿病数据集1.csv
│   └── 糖尿病数据集2.csv
├── models/                        # 模型文件（训练后生成）
│   ├── hypertension_model.pkl
│   ├── hypertension_scaler.pkl
│   ├── diabetes_model_general.pkl
│   ├── diabetes_scaler_general.pkl
│   ├── diabetes_model_pima.pkl
│   ├── diabetes_scaler_pima.pkl
│   └── avg_values.csv
├── src/
│   ├── train_model.py             # 模型训练脚本
│   ├── predict.py                 # 预测核心函数
│   └── llm_analyzer.py            # AI 大模型调用
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── picture/services.jpg
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   └── service.html
├── app.py                         # Flask 主应用
├── db_config.py                   # 数据库配置
├── requirements.txt               # 依赖列表
├── Dockerfile                     # Docker 镜像构建
├── docker-compose.yml             # Docker 编排
├── docker-entrypoint.sh           # 容器启动脚本
├── .dockerignore
├── .env.example
└── README.md
```

## 🚀 快速开始

### 方式一：本地运行

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置数据库
   确保本地 MySQL 已启动，创建数据库 `test_db`，修改 `db_config.py` 或设置环境变量：
```bash
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=你的密码
DB_NAME=test_db
```

3. 训练模型
```bash
python src/train_model.py
```

4. 配置 AI 密钥（可选）
```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-你的密钥"
# Linux/Mac
export DASHSCOPE_API_KEY="sk-你的密钥"
```

5. 启动服务
```bash
python app.py
```

6. 访问页面
打开浏览器访问：`http://localhost:5000`

### 方式二：Docker 一键部署（推荐）

1. 确保已安装 Docker 和 Docker Compose
2. 复制环境变量文件并填写配置
```bash
cp .env.example .env
# 编辑 .env 填入数据库密码和 AI 密钥
```

3. 启动
```bash
docker-compose up -d
```

4. 查看日志
```bash
docker-compose logs -f app
```

5. 访问 `http://localhost:5000`

## 🔧 核心算法

### 模型训练流程

数据清洗 → 特征工程 → 标准化 → 逻辑回归训练 → 超参数调优 → 模型保存

### 特征工程

- **交叉特征**：年龄×收缩压、BMI×血糖、血糖×糖化血红蛋白、高血压史×吸烟史
- **多项式特征**：收缩压²、年龄²、血糖²、BMI²
- **文本编码**：吸烟史（never→0, No Info→1, current→2, former→3）
- **计算特征**：BMI = 体重 / (身高/100)²

### 超参数调优

使用网格搜索遍历 C 值 `[0.01, 0.1, 1, 10, 100]`，结合 5 折交叉验证选择最优正则化强度。

### 模型准确率

| 模型 | 特征维度 | 测试集准确率 |
|------|---------|-------------|
| 高血压模型 | 7 → 14 | 73.5% |
| 糖尿病专业模型 | 6 → 12 | 82.3% |
| 糖尿病通用模型 | 6 → 13 | 81.1% |

### 智能模型选择

系统根据用户是否提供糖化血红蛋白以及高血压史、吸烟史、BMI 三项中非零项的数量，自动判断调用专业模型还是通用模型。

## 📊 数据流

用户输入 → Flask 接收 → 智能模型选择 → 缺失值填充（人群平均值） → 标准化 → 逻辑回归预测 → 生成 JSON → 前端渲染 → Plotly 图表 → AI 报告生成 → 报告导出

## 📈 性能指标

| 指标 | 数值 |
|------|------|
| 模型预测准确率 | 73.5% ~ 82.3% |
| API 响应时间 | < 100ms |
| 模型推理时间 | < 1ms/条 |
| 页面加载时间 | < 1s |
| 数据库查询 | < 50ms |

## 🔌 API 接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/register` | POST | 用户注册 |
| `/api/login` | POST | 用户登录 |
| `/api/logout` | POST | 退出登录 |
| `/api/predict` | POST | 健康风险评估 |
| `/api/user/info` | GET | 获取用户信息 |

## 📝 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DASHSCOPE_API_KEY` | 千问大模型 API 密钥 | 空 |
| `DB_HOST` | 数据库主机 | `127.0.0.1` |
| `DB_PORT` | 数据库端口 | `3306` |
| `DB_USER` | 数据库用户 | `root` |
| `DB_PASSWORD` | 数据库密码 | `111111` |
| `DB_NAME` | 数据库名称 | `test_db` |
| `APP_PORT` | 应用端口 | `5000` |

## 🧪 运行测试

```bash
# 训练模型
python src/train_model.py

# 预测测试
python src/predict.py

# 启动服务
python app.py
```

## 📄 开源协议

本项目仅供学习交流使用，采用 MIT 许可证。

## 🙏 致谢

- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [pandas](https://pandas.pydata.org/) - 数据分析库
- [Flask](https://flask.palletsprojects.com/) - 轻量级 Web 框架
- [Plotly](https://plotly.com/) - 交互式可视化库
- [阿里云百炼](https://bailian.console.aliyun.com/) - 千问大模型 API

## 📧 联系方式

如有问题，请联系项目维护者。

⭐ 如果觉得这个项目对你有帮助，欢迎给个 Star！
