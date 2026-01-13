```markdown
# Hands-On-Learning-LangChain

# 🚀 动手学 LangChain

🤖 **《手把手实践 LangChain 官方文档：从核心组件到高级用法》**

本教程提供与 LangChain 官方文档结构同步的**代码驱动式实践指南**。我相信，理解框架的最佳方式不是阅读，而是动手构建。

---

## 🎯 项目介绍

LangChain 已成为构建大语言模型应用的事实标准框架。然而，其官方文档概念繁多，初学者常感无从下手。仅仅阅读文档，难以掌握如何将分散的组件组合成真正可用的应用。

**动手学 LangChain** 项目正是为了解决这一问题而生。我们不是另一份文档翻译或理论教程，而是一份 **"代码优先"的同步实践指南**。

本教程严格遵循 LangChain 官方文档的核心脉络，**手把手带你为每一个重要概念编写可运行的代码**。从 `pip install` 后的第一行代码，到智能体（Agents）、模型（Models）、记忆（Memorys）、工具（Tools)等核心组件，再到守卫（Guardrails）、多智能体系统等高级用法，你将通过一系列循序渐进的示例，在实践中完成对框架的深度理解。

我们的目标是：让你在通读官方文档时，手边永远有一份可运行、可修改、可调试的**代码参考**，真正从"知道"走向"做到"。

---


## ⚙️ 开始前的准备

在开始本教程之前，请确保你已经完成以下准备工作：

### 1. 安装 LangChain 和相关依赖

```bash
# 安装 LangChain 核心包
pip install langchain

# 安装 OpenAI 兼容包（用于调用 DeepSeek 等模型）
pip install langchain-openai

# 安装其他常用依赖
pip install langchain-community  # 社区维护的集成
pip install langchain-text-splitters  # 文本分割工具
pip install langchain-chroma  # 向量数据库
```

### 2. 配置 DeepSeek 模型

本教程的所有代码示例均使用 **DeepSeek** 模型进行演示。你需要：

#### 获取 API Key
1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册账号并登录
3. 在控制台中创建 API Key

#### 安装 DeepSeek SDK
```bash
# 安装 DeepSeek 官方 SDK
pip install deepseek-api

# 或者使用 OpenAI 兼容方式（推荐）
pip install openai
```

#### 设置环境变量
```bash
# Linux/Mac
export DEEPSEEK_API_KEY="你的API密钥"
export OPENAI_API_KEY="你的API密钥"  # 如果使用OpenAI兼容方式
export OPENAI_API_BASE="https://api.deepseek.com"  # DeepSeek API地址

# Windows
set DEEPSEEK_API_KEY=你的API密钥
set OPENAI_API_KEY=你的API密钥
set OPENAI_API_BASE=https://api.deepseek.com
```

### 3. 创建配置文件（可选）

在项目根目录创建 `.env` 文件：

```env
# .env
DEEPSEEK_API_KEY=你的API密钥
OPENAI_API_KEY=你的API密钥
OPENAI_API_BASE=https://api.deepseek.com
```

然后在 Python 代码中使用：
```python
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量
```

### 4. 验证安装

运行以下代码验证环境配置是否正确：

```python
import os
from langchain_openai import ChatOpenAI

# 初始化 DeepSeek 模型
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com"
)

# 测试调用
response = llm.invoke("你好，请用一句话介绍你自己")
print(response.content)
```

如果看到 DeepSeek 模型的回复，说明环境配置成功！

### 5. 其他实用工具（可选）

```bash
# 用于环境变量管理
pip install python-dotenv

# 用于数据可视化
pip install matplotlib seaborn

# 用于数据处理
pip install pandas numpy

# 用于异步编程
pip install asyncio aiohttp
```

---

## ✨ 你将收获什么？

- 🛠️ **肌肉记忆般的组件掌握**：对 LangChain 核心抽象（模型、提示、链、智能体、记忆）的理解不止于概念，而是能信手写出的代码。
- 📖 **官方文档的最佳伴侣**：获得与官方文档每个核心章节对应的可运行代码示例，学习效率倍增。
- 🔗 **从零件到整机的能力**：掌握将零散组件组装成功能链（Chain），进而构建自主智能体（Agent）的系统方法。
- ⚙️ **解锁高级生产技能**：深入实践上下文工程、守卫、模型上下文协议（MCP）等构建可靠应用所必需的高级特性。

---

## 📖 内容导航：与官方文档同步实践

本教程内容模块与 LangChain 官方核心概念紧密对应，确保你学到的每个技能都能直接应用于项目开发。

| 模块         | 核心章节                      | 关键实践内容                         | 状态 |
| :----------- | :---------------------------- | :----------------------------------- | :------- |
| **核心组件** | [第一章 智能体](https://github.com/codeSnakeKun/Hands-On-Learning-LangChain/blob/main/docs/chapter1/%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6--%E6%99%BA%E8%83%BD%E4%BD%93%E5%AE%9E%E8%B7%B5.md) | LLM 智能体在循环中运行工具以实现目标 | ✅
|              | 第二章 模型                   | -                                    | 🚧       |
|              | 第三章 消息                   | -                                    | 🚧       |
|              | 第四章 工具                   | -                                    | 🚧       |
|              | 第五章 短期记忆               | -                                    | 🚧       |
|              | 第六章 流式传输               | -                                    | 🚧       |
|              | 第七章 中间件                 | -                                    | 🚧       |
|              | 第八章 结构化输出             | -                                    | 🚧       |
| **高级用法** | 第九章 守卫                   | -                                    | 🚧       |
|              | 第十章 运行时                 | -                                    | 🚧       |
|              | 第十一章 智能体中的上下文工程 | -                                    | 🚧       |
|              | 第十一章 模型上下文协议 (MCP) | -                                    | 🚧       |
|              | 第十二章 人机交互             | -                                    | 🚧       |
|              | 第十三章 多智能体系统         | -                                    | 🚧       |
|              | 第十四章 检索                 | -                                    | 🚧       |
|              | 第十五章 长期记忆             | -                                    | 🚧       |

---

## 💡 学习方法论：如何高效使用本教程

**1. 对照学习，效果更佳**
我们强烈建议你**同时打开官方文档和本教程的对应章节**。先阅读官方文档理解概念，然后立刻运行、阅读并修改本教程的配套代码。这种"概念-实践"的即时反馈循环是最高效的学习方式。

**2. 代码优先，动手至上**
本教程的所有知识点都凝结在代码示例中。不要满足于"能运行"，要敢于修改参数、拆解链条、触发错误，观察会发生什么。真正的理解源于调试。

**3. 模块化推进，积累成就感**
教程按模块组织，每个模块的代码都是自包含的。你可以按顺序系统学习，也可以根据项目需要，直接切入特定模块（如"检索"或"智能体"）。每完成一个模块，你都将获得一个可直接复用的代码库。

**面向读者**：本教程假设你已掌握Python基础，并对大语言模型有基本概念。无论你是希望快速上手的应用开发者，还是寻求深入理解框架细节的研究者，这里都有适合你的内容。

---

## 🤝 社区共建

一个开源教程的生命力源于社区的贡献。我们期待你的参与：

- **反馈与建议**：如果你在实践过程中遇到困惑，或对教程内容有改进想法，欢迎在 [GitHub Issues](https://github.com/codeSnakeKun/Hands-On-Learning-LangChain/issues) 提出。
- **代码贡献**：如果你发现了bug，或希望补充更优雅的示例，请查阅 [贡献指南](CONTRIBUTING.md) 并提交Pull Request。
- **知识分享**：基于本教程构建了有趣的项目？欢迎在 Discussions 中分享你的经验，激励更多人。
- **星标支持**：如果这个项目对你有帮助，请点击页面右上角的 **⭐ Star**，让更多人发现它。

## 📄 许可证

本项目所有教程内容与代码示例均采用 [MIT 许可证](LICENSE) 开源。

---

**现在，打开你的编辑器，开始动手吧！**
```
