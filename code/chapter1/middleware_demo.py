# LangChain智能体完整综合示例（DeepSeek）
# 本示例涵盖教程中的所有核心概念

# 导入必要的库
from langchain.agents import create_agent, AgentState  # 用于创建智能体和状态
from langchain_deepseek import ChatDeepSeek  # DeepSeek模型集成
from langchain.tools import tool  # 用于定义工具
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse  # 用于动态模型选择
from langchain.agents.structured_output import ToolStrategy  # 用于结构化输出
from pydantic import BaseModel  # 用于定义数据模型
from dotenv import load_dotenv  # 用于加载环境变量
import os  # 用于访问环境变量
from typing import TypedDict  # 用于类型化字典

# 加载环境变量（从.env文件中读取）
load_dotenv()

# 确保API密钥已加载
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("错误：DEEPSEEK_API_KEY 环境变量未设置")
    exit(1)  # 如果API密钥未设置，退出程序

# 1. 定义工具
print("=== 1. 定义工具 ===")


# 搜索工具
@tool
def search(query: str) -> str:
    """
    搜索信息

    参数：
    - query: str类型，搜索查询词

    返回值：
    - str类型，搜索结果
    """
    return f"搜索结果：{query} - 这是模拟的搜索结果"


# 天气工具
@tool
def get_weather(location: str) -> str:
    """
    获取位置的天气信息

    参数：
    - location: str类型，位置名称

    返回值：
    - str类型，天气信息
    """
    return f"{location} 的天气：晴朗，25°C"


# 2. 定义结构化输出模型
print("\n=== 2. 定义结构化输出模型 ===")


class ContactInfo(BaseModel):
    """联系信息模型"""
    name: str  # 姓名
    email: str  # 邮箱
    phone: str  # 电话


# 3. 定义自定义状态
print("\n=== 3. 定义自定义状态 ===")


class CustomState(AgentState):
    """自定义智能体状态"""
    user_preferences: dict  # 用户偏好设置


# 4. 创建动态模型选择中间件
print("\n=== 4. 创建动态模型选择中间件 ===")

# 创建模型实例
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=api_key
)


# 动态模型选择中间件
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """
    根据对话复杂性选择模型

    参数：
    - request: ModelRequest类型，包含请求信息
    - handler: 处理函数

    返回值：
    - ModelResponse类型，模型响应
    """
    message_count = len(request.state["messages"])
    print(f"消息数量: {message_count}")

    # 这里可以根据需要选择不同的模型
    # 目前我们只有一个模型，所以直接使用
    request.model = model
    return handler(request)


# 5. 创建智能体
print("\n=== 5. 创建智能体 ===")

agent = create_agent(
    model=model,
    tools=[search, get_weather],
    middleware=[dynamic_model_selection],
    system_prompt="你是一个有帮助的助手。请简洁准确地回答问题。"
)

print("智能体创建完成！")

# 6. 测试函数
print("\n=== 6. 开始测试 ===")


def test_agent():
    print("=== LangChain智能体综合测试（DeepSeek模型）===")

    # 测试1：天气查询
    print("\n测试1：天气查询")
    print("用户：上海今天天气怎么样？")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "上海今天天气怎么样？"}]}
    )
    print(f"智能体：{result1['messages'][-1].content}")

    # 测试2：信息搜索
    print("\n测试2：信息搜索")
    print("用户：2024年世界杯冠军是谁？")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "2024年世界杯冠军是谁？"}]}
    )
    print(f"智能体：{result2['messages'][-1].content}")

    # 测试3：流式传输
    print("\n测试3：流式传输")
    print("用户：搜索人工智能最新进展")
    print("正在流式获取响应...")
    print()

    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "搜索人工智能最新进展"}]
    }, stream_mode="values"):
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            print(f"智能体：{latest_message.content}")
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            print(f"正在调用工具：{[tc['name'] for tc in latest_message.tool_calls]}")
        print()


# 7. 运行测试
if __name__ == "__main__":
    test_agent()
    print("\n所有测试完成！")