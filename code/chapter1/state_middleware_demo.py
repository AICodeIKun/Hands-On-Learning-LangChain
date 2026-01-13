# 通过中间件定义状态示例（DeepSeek）
# 本示例演示如何通过中间件定义和使用自定义状态

# 导入必要的库
from langchain.agents import AgentState  # 智能体状态基类
from langchain.agents.middleware import AgentMiddleware  # 智能体中间件基类
from langchain.agents import create_agent  # 用于创建智能体
from langchain_deepseek import ChatDeepSeek  # DeepSeek模型集成
from langchain.tools import tool  # 用于定义工具
from dotenv import load_dotenv  # 用于加载环境变量
import os  # 用于访问环境变量
from typing import Any  # 用于类型提示

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


# 创建自定义中间件
# 继承AgentMiddleware，定义自定义状态和工具
class CustomMiddleware(AgentMiddleware):
    """自定义中间件，用于处理自定义状态"""
    state_schema = CustomState  # 指定状态模式
    tools = [get_recommendation]  # 关联的工具

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        """
        模型调用前的处理

        参数：
        - state: CustomState类型，当前状态
        - runtime: 运行时信息

        返回值：
        - 可选的状态更新
        """
        # 打印用户偏好
        print(f"用户偏好: {state.get('user_preferences', {})}")
        # 可以在这里修改状态或添加额外信息
        return None


print("自定义中间件创建完成，开始创建模型和智能体...")

# 创建ChatDeepSeek模型实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

# 创建智能体，传入模型和中间件
agent = create_agent(
    model=model,  # 模型实例
    tools=[],  # 工具将通过中间件添加
    middleware=[CustomMiddleware()]  # 传入自定义中间件
)

print("智能体创建完成，开始测试自定义状态...")

# 测试自定义状态
if __name__ == "__main__":
    print("=== 测试：自定义状态 ===")
    # 调用智能体，传入自定义状态
    result = agent.invoke({
        "messages": [{"role": "user", "content": "推荐一些电影"}],
        "user_preferences": {"style": "technical", "verbosity": "detailed"},  # 传入用户偏好
    })

    # 打印智能体的回复
    print("智能体回复:", result["messages"][-1].content)

    # 分析响应结构
    print("\n=== 响应结构分析 ===")
    print(f"响应包含的键: {list(result.keys())}")
    if 'user_preferences' in result:
        print(f"返回的用户偏好: {result['user_preferences']}")