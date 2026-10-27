# AMA4602 Kaggle Midterm Skill

把「读题 → 数据检查 → 防泄漏验证 → 建模 → CSV 校验 → 报告」变成可复用的 AI agent skill，供 AMA4602 同学备考和在课程明确允许的场景中使用。

**当前版本：0.1.0。** 仓库根目录就是 `ama4602-kaggle-exam` skill；只需一个副本，无须安装上游 skills。2026 考试安排和 PANCAN 内容是历史背景，实际任务、身份、截止时间和使用规则以当年的题目为准。

## 1. 下载与安装

需要 Git、Python **3.10 或以上**及能够读写本地文件、执行 Python 的 agent。macOS / Linux 可直接按以下步骤操作；Windows 可用 PowerShell，虚拟环境解释器改为 `.venv\Scripts\python.exe`。

```sh
git clone https://github.com/DumanHaoqian/AMA4602-Kaggle-Midterm.git
cd AMA4602-Kaggle-Midterm
python3 -B scripts/install_skill.py --client codex
```

安装选项：

| 客户端 / 方式 | 命令 / 目标位置 |
|---|---|
| Codex | `--client codex` → `~/.agents/skills/ama4602-kaggle-exam` |
| 较早的 Codex 环境 | `--client codex-legacy` → `$CODEX_HOME/skills/ama4602-kaggle-exam`，默认 `~/.codex/skills/ama4602-kaggle-exam` |
| Claude Code | `--client claude` → `~/.claude/skills/ama4602-kaggle-exam` |
| 项目内或其他兼容客户端 | `--dest /absolute/project/.agents/skills/ama4602-kaggle-exam`，参数是**完整目标文件夹** |

Codex 的当前发现目录见 [OpenAI 官方 skills 文档](https://developers.openai.com/codex/skills)。其他客户端需支持 `SKILL.md` 和本地脚本，具体发现方式以客户端版本为准。安装器仅复制文件，不修改客户端配置、安装依赖或注册内核。目标已存在时会停止；更新时先把旧目录移到安装目录之外备份，再重新安装。若列表未刷新，打开新的客户端会话。

也可以下载 GitHub Release ZIP，解压后进入包含 `SKILL.md` 的目录，运行同一安装命令。普通 GitHub “Source code” ZIP 和独立 skill ZIP 的外层文件夹名称不同，但内部入口相同。

在 Codex 对话中调用：

```text
$ama4602-kaggle-exam
请先阅读这个工作目录里的考试说明和数据，确认任务、指标和剩余时间，
然后完成本地验证、建模和提交文件检查。我的身份以当前 config.json 为准。
```

在 Claude Code 中使用 `/ama4602-kaggle-exam`，或明确让 agent 读取安装目录的 `SKILL.md`。普通网页聊天若不能访问本地目录、执行 Python，只能参考指引，无法直接运行本工具。

## 2. 准备 Python 环境

在刚下载的仓库目录执行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

以下命令均以这个虚拟环境为例。告诉 agent 使用虚拟环境里的解释器，并加 `-B`；无需向全局 Python 安装依赖。

| 需要的功能 | 额外安装 |
|---|---|
| 数据检查、数值基线、混合表线性模型、CSV 校验 | `requirements.txt` 已足够 |
| 图表、HTML、LaTeX 源码、执行 Jupyter 报告 | `.venv/bin/python -m pip install -r requirements-reports.txt` |
| LightGBM / XGBoost / CatBoost | `.venv/bin/python -m pip install -r requirements-boosting.txt` |
| 维护仓库、运行静态发布检查 | `.venv/bin/python -m pip install -r requirements-dev.txt` |
| 编译 PDF | 另装 [Tectonic](https://tectonic-typesetting.github.io/)，首次编译需联网下载 LaTeX 包 |
| Kaggle 下载 / 上传 | 另装官方 Kaggle CLI，并自行在本机完成认证；先查看安装版本的 `--help` |

macOS 上的 boosting 库可能另需 OpenMP 运行库；基础功能可以先不装 boosting。没有可选库时，leaderboard 会记录跳过的模型，缺 LightGBM 时跳过对抗验证。实际执行情况以 `evidence.json` 为准。

依赖文件给出兼容范围，并非锁定环境；每次运行会记录实际版本。不要在正式限时考试中临时升级环境。

## 3. 先做一次离线演练

以下仅生成合成数据与虚构身份，不下载课程数据，也不访问 Kaggle：

```sh
.venv/bin/python -B scripts/create_demo.py --output-dir demo
.venv/bin/python -B scripts/profile_data.py --config demo/config.json
.venv/bin/python -B scripts/evaluate_models.py --config demo/config.json --models select_logistic --budget-seconds 60
```

记下上一条命令输出的实际 run 目录，用它替换下面的 `RUN_DIR`：

```sh
.venv/bin/python -B scripts/validate_submission.py --config demo/config.json --predictions RUN_DIR/test_predictions.csv --evidence RUN_DIR/evidence.json --output-dir demo/final/v1
```

结果为 `demo/final/v1/DEMO0000_Demo_Student.csv` 和本地收据。这份演练文件不能用于真实比赛。每次生成数据、导出提交文件和打包都拒绝覆盖现有目标；再次演练时使用新目录。

完整检查分类、回归、混合表、ID 对齐和覆盖保护：

```sh
.venv/bin/python -B scripts/smoke_test.py
# 已装报告依赖时，加上图表、HTML、执行 notebook 和 LaTeX 源码检查：
.venv/bin/python -B scripts/smoke_test.py --reports
```

检查使用临时合成数据，结束后自动清理。`--reports` 不编译 PDF，因此不触发 Tectonic 下载。

## 4. 用于自己的题目

在独立的考试工作目录放置数据，并从 `assets/exam_config.example.json` 复制一份 `config.json`。所有数据和输出路径相对于 **config.json 所在目录**解析。

必须填写或核对：

- `task`、`target`、`metric`：实际任务和官方评价指标；回归同时改为 `prediction.kind=regression` 和合适的 `split.method`。
- `id_column`、`labels_path`、`label_id_column`：数据与外置标签的连接方式，不能默认按位置拼接。
- `sample_submission_path`、`submission_id_column`、`prediction.columns`：按真实样例 CSV；概率任务另设 `class_order` / `positive_class`。
- `student_id`、`full_name`：当前使用者的真实身份；脚本按 `StudentID_FullName.csv` 输出，这是本 skill 的 AMA4602 文件名约定。其他课程若要求不同命名，需明确修改适配。
- `competition`、`deadline`、`split`、预算与允许使用的资源：以当前说明为准，默认 `deadline=null`，不会自动推断。

数值表先用 `evaluate_models.py`；包含类别列时改用 `leaderboard.py`：

```sh
.venv/bin/python -B scripts/leaderboard.py --config /absolute/exam/config.json --families logreg_l2 --max-trials 1 --repeats 1 --seeds 1 --no-adversarial --budget-seconds 90
```

上面是分类的基础安装示例；回归可把 family 换成 `ridge`。完整候选、指标和任务流程见 [SKILL.md](SKILL.md) 与 [references/leaderboard.md](references/leaderboard.md)。预算只在模型拟合边界检查，无法中断已开始的一次拟合，要给最终预测和提交留出时间。

`validate_submission.py` **只校验本地文件**。Kaggle 上传须在当前用户授权、比赛规则允许且认证可用时另外执行；本地校验通过不等于平台已经收分。不要把 `kaggle.json`、token、真实身份配置、课程数据或作答结果提交到本仓库。

## 5. 图表与报告

先安装报告依赖。`plot_results.py --run RUN_DIR` 从已完成的结果生成图，不重新训练。复制 `assets/report_minimal.md` 到考试工作目录，填入身份和当前 run/config 路径，重写说明并替换所有 `REPLACE_` 标记，然后：

```sh
.venv/bin/python -B scripts/build_reports.py --source /absolute/exam/reports/report.md --formats tex,html,ipynb --no-compile
```

去掉 `--no-compile` 后会尝试调用 Tectonic；未安装时只有 `.tex`，必须检查构建状态，不能声称已经生成 PDF。Notebook 用当前 Python 环境执行。HTML 内嵌 Plotly，图表交互可离线使用；数学排版和自定义字体使用 KaTeX / Google Fonts CDN，需要网络。详见 [报告说明](references/reporting.md)。

## 6. 发布与贡献

维护步骤见 [CONTRIBUTING.md](CONTRIBUTING.md)。发布前运行：

```sh
.venv/bin/python -B scripts/check_release.py
.venv/bin/python -B scripts/smoke_test.py --reports
.venv/bin/python -B scripts/build_release.py
```

生成 `dist/ama4602-kaggle-exam-v0.1.0.zip`，内部只有一个 `ama4602-kaggle-exam/` skill 文件夹。打包采用明确的文件清单，排除数据、输出、环境与凭据；不计算文件摘要。GitHub Actions 会在 push / PR 时检查，在 `v*` tag 时额外生成可下载的构建 artifact；**不会自动创建公开 GitHub Release**。维护者可将 ZIP 附加到 Release。

原创部分采用 [MIT](LICENSE)。NeurIPS 模板保留独立署名和许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；方法参考见 [sources.md](references/sources.md)。本项目为同学间的学习工具，与 PolyU、Kaggle、NeurIPS 无官方隶属关系。
