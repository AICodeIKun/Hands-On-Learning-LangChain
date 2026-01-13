# 工具错误处理示例（DeepSeek）
# 本示例演示如何处理工具执行过程中的错误

# 导入必要的库
from langchain.agents import create_agent  # 用于创建智能体
from langchain.agents.middleware import wrap_tool_call  # 用于创建工具调用中间件
from langchain_core.messages import ToolMessage  # 用于创建工具消息
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

# 定义一个会出错的工具
# 使用@tool装饰器定义一个除法工具
@tool
def divide(a: int, b: int) -> str:
    """
    执行除法运算
    
    参数：
    - a: int类型，被除数
    - b: int类型，除数
    
    返回值：
    - str类型，除法结果
    """
    # 当b为0时，会触发除零错误
    return str(a / b)

# 创建错误处理中间件
# 使用@wrap_tool_call装饰器创建工具调用中间件
@wrap_tool_call
def handle_tool_errors(request, handler):
    """
    使用自定义消息处理工具执行错误
    
    参数：
    - request: 工具调用请求
    - handler: 处理函数
    
    返回值：
    - 处理结果或错误消息
    """
    try:
        # 尝试调用处理函数
        return handler(request)
    except Exception as e:
        # 捕获异常，创建自定义错误消息
        print(f"捕获到工具错误: {str(e)}")
        # 创建ToolMessage，包含错误信息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]  # 确保传递正确的工具调用ID
        )

print("错误处理中间件创建完成，开始创建模型和智能体...")

# 创建ChatDeepSeek模型实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

# 创建智能体，传入模型、工具和错误处理中间件
agent = create_agent(
    model=model,  # 模型实例
    tools=[divide],  # 传入除法工具
    middleware=[handle_tool_errors]  # 传入错误处理中间件
)

print("智能体创建完成，开始测试错误处理...")

# 测试智能体
if __name__ == "__main__":
    # 测试错误情况
    print("=== 测试：除零错误处理 ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "计算 10 除以 0"}]}
    )
    print("智能体回复:", result["messages"][-1].content)
    
    # 测试正常情况
    print("\n=== 测试：正常除法运算 ===")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "计算 10 除以 2"}]}
    )
    print("智能体回复:", result2["messages"][-1].content)