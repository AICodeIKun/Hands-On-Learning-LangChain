# 通过state_schema定义状态示例（DeepSeek）
# 本示例演示如何通过state_schema参数定义和使用自定义状态

# 导入必要的库
from langchain.agents import AgentState  # 智能体状态基类
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


# 定义自定义状态
# 继承AgentState，添加user_preferences字段
class CustomState(AgentState):
    """自定义智能体状态"""
    user_preferences: dict  # 用户偏好设置


# 定义工具
# 使用@tool装饰器定义一个推荐工具
@tool
def get_recommendation(topic: str) -> str:
    """
    获取推荐信息

    参数：
    - topic: str类型，推荐主题

    返回值：
    - str类型，推荐信息
    """
    # 这里是模拟的推荐信息，实际应用中可以根据主题生成真实推荐
    return f"关于 {topic} 的推荐：这是为您精心挑选的内容。"


print("状态模式和工具定义完成，开始创建模型和智能体...")

# 创建ChatDeepSeek模型实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

# 创建智能体，传入模型、工具和状态模式
agent = create_agent(
    model=model,  # 模型实例
    tools=[get_recommendation],  # 传入推荐工具
    state_schema=CustomState  # 指定自定义状态模式
)

print("智能体创建完成，开始测试自定义状态...")

# 测试自定义状态
if __name__ == "__main__":
    print("=== 测试：自定义状态 ===")
    # 调用智能体，传入自定义状态
    result = agent.invoke({
        "messages": [{"role": "user", "content": "推荐一些书籍"}],
        "user_preferences": {"style": "technical", "verbosity": "detailed"},  # 传入用户偏好
    })

    # 打印智能体的回复
    print("智能体回复:", result["messages"][-1].content)

    # 分析响应结构
    print("\n=== 响应结构分析 ===")
    print(f"响应包含的键: {list(result.keys())}")
    if 'user_preferences' in result:
        print(f"返回的用户偏好: {result['user_preferences']}")