# 工具定义示例（DeepSeek）
# 本示例演示如何定义和使用工具

# 导入必要的库
from langchain.tools import tool  # 用于定义工具
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

# 定义工具
# 使用@tool装饰器定义一个搜索工具
@tool
def search(query: str) -> str:
    """
    搜索信息
    
    参数：
    - query: str类型，搜索查询词
    
    返回值：
    - str类型，搜索结果
    """
    # 这里是模拟的搜索结果，实际应用中可以集成真实的搜索引擎
    return f"结果：{query}"

# 使用@tool装饰器定义一个天气查询工具
@tool
def get_weather(location: str) -> str:
    """
    获取位置的天气信息
    
    参数：
    - location: str类型，位置名称
    
    返回值：
    - str类型，天气信息
    """
    # 这里是模拟的天气信息，实际应用中可以集成真实的天气API
    return f"{location} 的天气：晴朗，72°F"

print("工具定义完成，开始创建模型和智能体...")

# 创建ChatDeepSeek模型实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

# 创建智能体，传入模型和工具
agent = create_agent(
    model=model,  # 模型实例
    tools=[search, get_weather]  # 传入定义的工具列表
)

print("智能体创建完成，开始测试...")

# 测试智能体
if __name__ == "__main__":
    # 测试天气工具
    print("=== 测试1：天气查询 ===")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "北京的天气怎么样？"}]}
    )
    print("天气查询结果:", result1["messages"][-1].content)
    
    # 测试搜索工具
    print("\n=== 测试2：信息搜索 ===")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "2024年奥运会在哪里举行？"}]}
    )
    print("搜索结果:", result2["messages"][-1].content)