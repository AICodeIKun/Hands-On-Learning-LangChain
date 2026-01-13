# 智能体调用示例（DeepSeek）
# 本示例演示如何调用智能体并处理其响应

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
    # 改进搜索工具，使其能处理天气查询
    if "天气" in query or "weather" in query.lower():
        # 模拟天气搜索结果
        if "旧金山" in query or "san francisco" in query.lower():
            return "旧金山天气：今天晴朗，温度 18-22°C，湿度 65%，风速 15 km/h。"
        else:
            return f"搜索到 {query} 的天气信息：请查看当地天气预报。"
    else:
        return f"搜索结果：{query}"


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

print("智能体创建完成，开始测试调用...")

# 测试调用智能体
if __name__ == "__main__":
    print("=== 测试：基本调用 ===")
    # 基本调用
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "旧金山天气如何？"}]}
    )

    # 打印智能体的回复
    print("智能体回复:", result["messages"][-1].content)

    # 分析响应结构
    print("\n=== 响应结构分析 ===")
    print(f"响应类型: {type(result)}")
    print(f"响应包含的键: {list(result.keys())}")
    print(f"消息数量: {len(result['messages'])}")

    # 打印所有消息
    print("\n=== 所有消息 ===")
    for i, message in enumerate(result['messages']):
        print(f"消息 {i + 1}:")

        # 获取角色 - 处理不同的消息类型
        if hasattr(message, 'type'):
            role = message.type
        elif hasattr(message, 'role'):
            role = message.role
        else:
            role = type(message).__name__

        print(f"  角色: {role}")

        # 获取内容
        if hasattr(message, 'content') and message.content:
            print(f"  内容: {message.content}")
        else:
            print(f"  内容: 无")

        # 检查是否有工具调用
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"  工具调用: {message.tool_calls}")
        elif hasattr(message, 'tool_call_id'):
            print(f"  工具调用ID: {message.tool_call_id}")

        print()