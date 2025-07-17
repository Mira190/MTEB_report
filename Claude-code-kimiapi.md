# Claude Code 原生运行 + 接入 Kimi K2（Windows / macOS / Linux ｜无需 WSL）

## 一分钟速看：Claude Code 接入 Kimi api

1. 安装 Node.js ≥18（Windows 用 MSI，macOS 用 Homebrew）。
2. 验证安装：`node -v`、`npm -v`。
3. 安装 CLI：`npm install -g @anthropic-ai/claude-code`。
4. 初始化配置：运行 `claude` 后 Ctrl+C 退出。
5. 获取 Kimi Key：登录控制台 → 新建 Key → 复制 `sk-...`。
6. 设置环境变量：
   - 国内站：`ANTHROPIC_BASE_URL=https://api.moonshot.cn/anthropic/`
   - 国际站：`ANTHROPIC_BASE_URL=https://api.moonshot.ai/anthropic/`
   - 必填 Key：`ANTHROPIC_API_KEY=你的sk-...`
7. 编辑主目录下 `.claude.json`：

   ```json
   {
     "installMethod": "npm",
     "autoUpdates": false,
     "hasCompletedOnboarding": true,
     "telemetry": false,
     "customApiUrl": "https://api.moonshot.cn/anthropic/"
   }
   ```

完毕后新开终端测试：

```bash
claude --model kimi-k2-0711-preview
```

若模型正常响应，即配置成功（实际模型名以 Kimi 控制台可用列表为准）。

---

## 1. 为何用 Kimi 通道？

- **更省钱：** 调 Kimi 模型或兼容端点，成本通常低于直接用官方 Claude API（以当期计费为准）。
- **跨平台统一：** CLI 一套命令，Windows/macOS/Linux 通吃。
- **原生 Windows：** 无需 WSL、虚拟机。
- **代码工作流友好：** Claude Code 支持项目上下文、解释代码、生成/修改文件等编程场景。

---

## 2. 获取 Kimi API Key（CN / AI 区域）

1. 打开 Kimi 平台控制台：`platform.moonshot.cn/console/api`（
2. 登录或注册账号。
3. 进入 **API Key 管理 → 新建 API Key**。
4. 复制生成的 `sk-...`（关闭窗口后无法再查看；丢失可重建）。
5. 新账号一般会赠送一定试用金（目前是 15rmb）。

---

## 3. 平台安装 Node.js

Claude Code CLI 依赖 Node 环境。

### Windows（推荐 LTS MSI 安装）

1. 前往 Node.js 官网下载 LTS `.msi`（x64）。
2. 安装时保持默认，务必勾选 **Add to PATH**。
3. 安装完成后打开 **PowerShell** 验证：

   ```powershell
   node -v
   npm -v
   ```

### macOS（Homebrew 最方便）

```bash
brew update
brew install node
node -v && npm -v
```

### Linux（Ubuntu 示例；其他发行版见注）

**快速（发行版仓库版，可能不是最新）：**

```bash
sudo apt update
sudo apt install -y nodejs npm
```

## 4. 安装 Claude Code CLI

```bash
npm install -g @anthropic-ai/claude-code
```

初始化（生成默认配置文件）：

```bash
claude
```

出现欢迎/初始化输出后按 `Ctrl+C` 退出；用户主目录下应出现 `.claude.json`（若无，可手动创建）。

---

## 5. 选择区域 & 设置环境变量

Claude Code 通过环境变量读取 API 地址和密钥。你需要 **先决定使用 Kimi 国内（CN）还是国际（AI）端点**。

| 区域    | Base URL（填入 `ANTHROPIC_BASE_URL` & `.claude.json` → `customApiUrl`） |
| ------- | ----------------------------------------------------------------------- |
| CN 国内 | `https://api.moonshot.cn/anthropic/`                                    |
| AI 国际 | `https://api.moonshot.ai/anthropic/`                                    |

所有区域都需要：`ANTHROPIC_API_KEY=你的sk-...`

> 提示：Base URL 最后有无尾斜杠通常都可；若遇 404/402，先严格按上表复制再测。

### 5.1 临时变量（当前终端会话有效）

适合快速测试；关闭终端即失效。

**Windows PowerShell（CN 示例）**

```powershell
$env:ANTHROPIC_BASE_URL="https://api.moonshot.cn/anthropic/"
$env:ANTHROPIC_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

**切换国际站：**

```powershell
$env:ANTHROPIC_BASE_URL="https://api.moonshot.ai/anthropic/"
```

**CMD（CN 示例）**

```cmd
set ANTHROPIC_BASE_URL=https://api.moonshot.cn/anthropic/
set ANTHROPIC_API_KEY=sk-xxxxxxxxxxxxxxxx
```

国际站改 `.cn` → `.ai`。

**macOS / Linux（CN 示例）**

```bash
export ANTHROPIC_BASE_URL="https://api.moonshot.cn/anthropic/"
export ANTHROPIC_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

国际站：改为 `api.moonshot.ai`。

---

### 5.2 持久化变量

> 重启终端仍生效；以下均写入“当前用户”作用域。

##### Windows 命令行持久化（脚本）

```cmd
REM CN 国内站（默认）
setx ANTHROPIC_BASE_URL "https://api.moonshot.cn/anthropic/"
REM AI 国际站（在海外使用，将上一行注释掉并启用下行）
REM setx ANTHROPIC_BASE_URL "https://api.moonshot.ai/anthropic/"
setx ANTHROPIC_API_KEY "sk-xxxxxxxxxxxxxxxx"
```

> 提醒：`setx` 写入注册表，对新开的终端生效；当前窗口不更新。

#### Windows（图形界面）

1. 开始菜单搜索 **“查看高级系统设置”**（或 `Win+R` → `SystemPropertiesAdvanced`）。
2. _系统属性_ → _高级_ → 【环境变量…】。
3. 在 _用户变量_ 区域点【新建…】：

   - 变量名：`ANTHROPIC_BASE_URL`
   - 变量值（选一）：

     - 国内：`https://api.moonshot.cn/anthropic/`
     - 国际：`https://api.moonshot.ai/anthropic/`

   - 确定。

4. 再点【新建…】：

   - 变量名：`ANTHROPIC_API_KEY`
   - 变量值：你的 `sk-...`

5. 确认保存，关闭所有窗口。
6. 新开 PowerShell 验证：

   ```powershell
   echo $env:ANTHROPIC_BASE_URL
   echo $env:ANTHROPIC_API_KEY   # 小心暴露
   ```

**PowerShell .NET API 写入**

```powershell
# CN 国内站
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL","https://api.moonshot.cn/anthropic/","User")
# AI 国际站（取消注释使用）
# [System.Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL","https://api.moonshot.ai/anthropic/","User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY","sk-xxxxxxxxxxxxxxxx","User")
```

#### macOS/Linux

```bash
# CN 国内站（默认）
echo 'export ANTHROPIC_BASE_URL="https://api.moonshot.cn/anthropic/"' >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY="sk-xxxxxxxxxxxxxxxx"' >> ~/.zshrc
# 如在海外 / 代理环境使用国际站：
echo '# export ANTHROPIC_BASE_URL="https://api.moonshot.ai/anthropic/"' >> ~/.zshrc
source ~/.zshrc
```

若你使用 bash，请写入 `~/.bash_profile` 或 `~/.bashrc`。

## 6. 编辑 `.claude.json`

Claude Code CLI 会在首次运行后在用户主目录生成一个 `.claude.json`（若未生成，可手动创建）。你需要将 `customApiUrl` 指向你所选区域的 Base URL。

**最小工作示例（CN 国内站）：**

```json
{
  "installMethod": "npm",
  "autoUpdates": false,
  "hasCompletedOnboarding": true,
  "telemetry": false,
  "customApiUrl": "https://api.moonshot.cn/anthropic/"
}
```

**国际站改为：**`https://api.moonshot.ai/anthropic/`。

> 若你需要在 CLI 中频繁切换，可维护两个文件 `.claude.cn.json` / `.claude.ai.json` 然后复制覆盖 `.claude.json`。

---

## 7. 启动并验证

在你要与 Claude Code 协作的项目目录中运行：

```bash
claude
```

进入交互界面后，可指定模型：

```bash
/model kimi-k2-0711-preview
```

或直接命令行：

```bash
claude --model kimi-k2-0711-preview
```

> 模型名会因平台更新；请在 Kimi 控制台查看当前可用模型（如 `kimi-k2`, `kimi-k2-0711-preview` 等）。

如果调用成功，你可以在 Kimi 控制台的调用明细中看到请求记录。

---

## 8. 常见错误排查

### 402 错误（Payment Required / 无额度）

- 多见于 **Base URL 写错**（打漏字符、错域名）。
- 也可能为 **账号无可用余额**。
- 检查 `.claude.json` 与环境变量是否一致（CN vs AI）。

### 429 错误（Rate Limit Reached / RPM 超限）

你遇到的报错如下示例：

```
API Error: 429 {"error":{"message":"Your account org-... request reached organization max RPM: 3, please try again after 1 seconds","type":"rate_limit_reached_error"}}
```

含义：**在单位时间内的请求数超出组织级速率限制（Requests Per Minute）**。
处理建议：

- 减慢调用节奏（加 sleep / 重试退避）。
- 控制并发；避免短时间内大量补发重试。
- 如果脚本循环中频繁发请求，考虑批量上下文或流式交互减少次数。
- 若长期需要更高速率，联系平台或升级配额（视 Kimi / 账户政策）。

> “try again after 1 seconds” 是服务器建议的最小等待时间；实际可适当指数回退（1s→2s→4s）。

### 其它网络问题

- 国内网络访问国际站可能不稳；优先使用 CN 域。
- 代理环境下注意 NO_PROXY / HTTPS_PROXY 变量传递。

---

## 9. 快速命令速查

**检查 Node / NPM**

```bash
node -v
npm -v
```

**装 CLI**

```bash
npm install -g @anthropic-ai/claude-code
```

**临时环境变量（PowerShell CN）**

```powershell
$env:ANTHROPIC_BASE_URL="https://api.moonshot.cn/anthropic/"
$env:ANTHROPIC_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

**持久化（Windows cmd CN）**

```cmd
setx ANTHROPIC_BASE_URL "https://api.moonshot.cn/anthropic/"
setx ANTHROPIC_API_KEY "sk-xxxxxxxxxxxxxxxx"
```

**启动 + 指定模型**

```bash
claude --model kimi-k2-0711-preview
```
