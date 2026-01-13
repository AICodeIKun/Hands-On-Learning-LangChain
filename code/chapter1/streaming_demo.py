# 流式传输示例（DeepSeek）
# 本示例演示如何使用流式传输获取智能体的实时响应

# 导入必要的库
from langchain.agents import create_agent  # 用于创建智能体
from langchain_deepseek import ChatDeepSeek  # DeepSeek模型集成
from langchain.tools import tool  # 用于定义工具
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
    return f"搜索结果：{query} - 这是模拟的搜索结果"


print("工具定义完成，开始创建模型和智能体...")

# 创建ChatDeepSeek模型实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

# 创建智能体，传入模型和工具
agent = create_agent(
    model=model,  # 模型实例
    tools=[search]  # 传入搜索工具
)

print("智能体创建完成，开始测试流式传输...")

# 测试流式传输
if __name__ == "__main__":
    print("=== 测试：流式传输 ===")
    print("正在搜索AI新闻并总结...")
    print()

    # 使用stream方法进行流式传输
    # stream_mode="values"表示每个块包含该时间点的完整状态
    for chunk in agent.stream({
        "messages": [{"role": "user", "content": "搜索人工智能最新进展并总结发现"}]
    }, stream_mode="values"):
        # 每个块包含该时间点的完整状态
        latest_message = chunk["messages"][-1]  # 获取最新消息

        # 根据消息类型处理
        if latest_message.content:
            # 文本消息
            print(f"智能体：{latest_message.content}")
        elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
            # 工具调用消息
            print(f"正在调用工具：{[tc['name'] for tc in latest_message.tool_calls]}")
        elif hasattr(latest_message, 'tool_call_id'):
            # 工具响应消息
            print(f"工具响应收到")
        print()

    print("流式传输完成！")