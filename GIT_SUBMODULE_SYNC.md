# Git 子仓库更新同步指南

本仓库使用 Git Submodule 管理以下子仓库：


| 子仓库                     | 本地路径                                   | 跟踪分支 |
| ---------------------------- | -------------------------------------------- | ---------- |
| LLM-Resume-Template-Brench | `brench_resume`                            | `main`   |
| AURA-GRPO                  | `project_review/AURA-GRPO`                 | `main`   |
| Coding-Agent-SFT-Demo      | `project_review/Coding-Agent-SFT-Demo`     | `main`   |
| LLM-Code-Whiteboard        | `basic_knowledge/code/LLM-Code-Whiteboard` | `main`   |
| Policy-Query-Planner       | `project_review/Policy-Query-Planner`      | `main`   |

主仓库只记录每个子仓库的某个 Git 提交，不会自动跟随子仓库的最新提交。

## 1. 首次克隆本仓库

推荐在克隆时直接拉取全部子仓库：

```bash
git clone --recurse-submodules https://github.com/BrenchCC/Brench-Interview.git
cd Brench-Interview
```

如果已经克隆，但子仓库目录为空：

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## 2. 日常拉取主仓库

拉取主仓库，并将子仓库恢复到主仓库记录的提交：

```bash
git pull --recurse-submodules
git submodule sync --recursive
git submodule update --init --recursive
```

这是最稳妥的日常同步方式，不会主动把子仓库推进到远程最新提交。

## 3. 更新全部子仓库到 `main` 最新提交

先确认自己没有在子仓库中留下未提交修改：

```bash
git submodule foreach --recursive 'git status --short --branch'
```

然后拉取 `.gitmodules` 配置的远程分支：

```bash
git submodule sync --recursive
git submodule update --init --remote --merge --recursive
```

检查更新结果：

```bash
git submodule status --recursive
git status
```

子仓库更新后，主仓库会显示对应路径发生变化。需要在主仓库提交新的子仓库指针：

```bash
git add brench_resume
git add project_review/AURA-GRPO
git add project_review/Coding-Agent-SFT-Demo
git add basic_knowledge/code/LLM-Code-Whiteboard
git add project_review/Policy-Query-Planner
git commit -m "chore: update git submodules"
git push
```

只更新了部分子仓库时，只需 `git add` 实际发生变化的路径。

## 4. 只更新一个子仓库

以 `LLM-Code-Whiteboard` 为例：

```bash
git submodule update --init --remote --merge basic_knowledge/code/LLM-Code-Whiteboard
git add basic_knowledge/code/LLM-Code-Whiteboard
git commit -m "chore: update LLM-Code-Whiteboard submodule"
git push
```

## 5. 修改子仓库中的代码

必须先在子仓库内提交并推送代码，再回到主仓库提交子仓库指针。

```bash
cd basic_knowledge/code/LLM-Code-Whiteboard
git switch main
git pull --ff-only

# 修改文件后，在子仓库中提交
git add <修改的文件>
git commit -m "<子仓库提交信息>"
git push

# 返回主仓库
cd ../../..
git add basic_knowledge/code/LLM-Code-Whiteboard
git commit -m "chore: update LLM-Code-Whiteboard submodule"
git push
```

不要只在主仓库执行 `git add`：主仓库只能记录子仓库提交指针，不能代替子仓库提交其内部文件。

## 6. 新增一个子仓库

在主仓库根目录执行：

```bash
git submodule add -b main ../<子仓库名称> <本地路径>
git add .gitmodules <本地路径>
git commit -m "chore: add <子仓库名称> submodule"
git push
```

例如添加 `Policy-Query-Planner`：

```bash
git submodule add -b main ../Policy-Query-Planner project_review/Policy-Query-Planner
git add .gitmodules project_review/Policy-Query-Planner
git commit -m "chore: add Policy-Query-Planner submodule"
git push
```

其他人拉取主仓库后，执行以下命令即可初始化这个新子仓库：

```bash
git pull --recurse-submodules
git submodule sync --recursive
git submodule update --init --recursive
```

## 7. 常见问题

### 子仓库目录为空

```bash
git submodule update --init --recursive
```

### `.gitmodules` 的 URL 或分支发生变化

```bash
git submodule sync --recursive
git submodule update --init --remote --merge --recursive
```

### 子仓库显示 `HEAD detached`

主仓库按指定提交检出子仓库时，出现 detached HEAD 是正常现象。如果需要在该子仓库中开发，手动切回 `main`：

```bash
cd <子仓库路径>
git switch main
git pull --ff-only
```

### 更新前发现子仓库有未提交修改

先进入对应子仓库提交或暂存修改，再执行更新：

```bash
cd <子仓库路径>
git status
git stash
```

恢复暂存内容：

```bash
git stash pop
```

## 常用命令速查

### 修改并提交子仓库

先进入子仓库，拉取最新代码，再修改和提交：

```bash
cd <子仓库路径>
git switch main
git pull --ff-only

# 修改完成后
git status
git add <修改的文件>
git commit -m "<提交信息>"
git push
```

回到主仓库，提交更新后的子仓库指针：

```bash
cd <主仓库路径>
git add <子仓库路径>
git commit -m "chore: update <子仓库名称> submodule"
git push
```

### 拉取主仓库并同步到指定版本

按主仓库指定版本同步：

```bash
git pull --recurse-submodules
git submodule sync --recursive
git submodule update --init --recursive
```

### 主动拉取所有子仓库的最新版本

更新全部子仓库到远程最新版本：

```bash
git submodule sync --recursive
git submodule update --init --remote --merge --recursive
git status
```

确认变化后，在主仓库提交子仓库指针：

```bash
git add <发生变化的子仓库路径>
git commit -m "chore: update git submodules"
git push
```

### 查看同步状态

查看全部子仓库状态：

```bash
git submodule status --recursive
git submodule foreach --recursive 'git status --short --branch'
```
