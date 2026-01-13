# 动态系统提示示例（DeepSeek）
# 本示例演示如何根据用户角色动态生成系统提示

# 导入必要的库
from typing import TypedDict  # 用于定义类型化字典
from langchain.agents import create_agent  # 用于创建智能体
from langchain.agents.middleware import dynamic_prompt, ModelRequest  # 用于创建动态提示中间件
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


# 定义上下文类型
# 使用TypedDict定义上下文结构，包含user_role字段
class Context(TypedDict):
    user_role: str  # 用户角色


# 创建动态提示中间件
# 使用@dynamic_prompt装饰器创建动态提示中间件
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """
    根据用户角色生成系统提示

    参数：
    - request: ModelRequest类型，包含请求信息

    返回值：
    - str类型，生成的系统提示
    """
    # 从请求的运行时上下文中获取用户角色，处理context为None的情况
    context = request.runtime.context
    if context is None:
        user_role = "user"  # 默认为user
    else:
        user_role = context.get("user_role", "user")  # 从上下文中获取用户角色，默认为"user"

    # 基础提示
    base_prompt = "你是一个有帮助的助手。"

    # 根据用户角色生成不同的系统提示
    if user_role == "expert":
        # 专家角色：提供详细的技术响应
        return f"{base_prompt} 请提供详细的技术响应，包括深入的解释和专业术语。"
    elif user_role == "beginner":
        # 新手角色：简单解释概念，避免使用行话
        return f"{base_prompt} 请简单解释概念，避免使用行话，使用通俗易懂的语言。"
    else:
        # 默认角色：使用基础提示
        return base_prompt


print("动态提示中间件创建完成，开始创建模型和智能体...")

# 创建ChatDeepSeek模型实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

# 创建智能体，传入模型、中间件和上下文模式
agent = create_agent(
    model=model,  # 模型实例
    tools=[],  # 暂时为空工具列表
    middleware=[user_role_prompt],  # 传入动态提示中间件
    context_schema=Context  # 传入上下文模式
)

print("智能体创建完成，开始测试不同角色的系统提示...")

# 测试智能体
if __name__ == "__main__":
    # 测试专家角色
    print("=== 测试1：专家角色 ===")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "解释机器学习的原理"}]},
        context={"user_role": "expert"}  # 传入专家角色上下文
    )
    print("专家模式回复:", result1["messages"][-1].content)

    # 测试新手角色
    print("\n=== 测试2：新手角色 ===")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "解释机器学习的原理"}]},
        context={"user_role": "beginner"}  # 传入新手角色上下文
    )
    print("新手模式回复:", result2["messages"][-1].content)

    # 测试默认角色
    print("\n=== 测试3：默认角色 ===")
    result3 = agent.invoke(
        {"messages": [{"role": "user", "content": "解释机器学习的原理"}]}
        # 不传入上下文，使用默认角色
    )
    print("默认模式回复:", result3["messages"][-1].content)