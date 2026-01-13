# 系统提示示例（DeepSeek）
# 本示例演示如何使用系统提示指导智能体的行为

# 导入必要的库
from langchain.agents import create_agent  # 用于创建智能体
from langchain_deepseek import ChatDeepSeek  # DeepSeek模型集成
from dotenv import load_dotenv  # 用于加载环境变量
import os  # 用于访问环境变量

# 加载环境变量（从.env文件中读取）
load_dotenv()

# 确保API密钥已加载
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("错误：DEEPSEEK_API_KEY 环境变量未设置")
    exit(1)  # 如果API密钥未设置，退出程序

print("开始创建模型和智能体...")

# 创建ChatDeepSeek模型实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

# 创建带系统提示的智能体
# 系统提示用于指导智能体的行为和回答风格
agent = create_agent(
    model=model,  # 模型实例
    tools=[],  # 暂时为空工具列表
    system_prompt="你是一个有帮助的助手。请简洁准确地回答问题，不要添加多余的信息。"  # 系统提示
)

print("智能体创建完成，开始测试...")

# 测试智能体
if __name__ == "__main__":
    print("=== 测试：系统提示效果 ===")
    # 测试系统提示对智能体回答风格的影响
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "什么是LangChain？请详细解释。"}]}
    )
    print("智能体回复:", result["messages"][-1].content)

    # 测试另一个问题
    print("\n=== 测试：另一个问题 ===")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "如何使用Python创建一个简单的函数？"}]}
    )
    print("智能体回复:", result2["messages"][-1].content)