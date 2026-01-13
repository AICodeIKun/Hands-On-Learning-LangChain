# 动态模型示例（DeepSeek）
# 本示例演示如何根据对话复杂性动态选择模型

# 导入必要的库
from langchain_deepseek import ChatDeepSeek  # DeepSeek模型集成
from langchain.agents import create_agent  # 用于创建智能体
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse  # 用于创建中间件
from dotenv import load_dotenv  # 用于加载环境变量
import os  # 用于访问环境变量

# 加载环境变量（从.env文件中读取）
load_dotenv()

# 确保API密钥已加载
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("错误：DEEPSEEK_API_KEY 环境变量未设置")
    exit(1)  # 如果API密钥未设置，退出程序

# 创建模型实例
basic_model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

advanced_model = ChatDeepSeek(
    model="deepseek-reasoner",
    api_key=api_key
)

# 创建动态模型选择中间件
# 使用@wrap_model_call装饰器创建中间件，用于修改请求中的模型
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """
    根据对话复杂性选择模型
    
    参数：
    - request: ModelRequest类型，包含请求的状态和上下文
    - handler: 处理函数，用于处理修改后的请求
    
    返回值：
    - ModelResponse类型，包含模型的响应
    """
    # 获取当前对话中的消息数量，用于判断对话复杂性
    message_count = len(request.state["messages"])
    print(f"当前对话消息数量: {message_count}")

    # 根据消息数量选择不同的模型处理方式
    if message_count > 10:
        # 对较长的对话，使用推理模型
        model = basic_model
        print("使用高级模型处理: deepseek-reasoner")
    else:
        # 对较短的对话，使用基础模型
        model = basic_model
        print("使用基础模型处理: deepseek-chat")

    # 将选择的模型设置到请求中
    request.model = model
    # 调用处理函数处理请求
    return handler(request)

# 创建智能体
# 传入默认模型和中间件
agent = create_agent(
    model=basic_model,  # 默认模型
    tools=[],  # 暂时为空工具列表
    middleware=[dynamic_model_selection]  # 传入动态模型选择中间件
)

# 测试智能体
if __name__ == "__main__":
    print("=== 测试1：基本对话（消息数量少）===")
    # 测试基本对话（消息数量少）
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，介绍一下自己"}]}
    )
    print("智能体回复:", result["messages"][-1].content)
    
    print("\n=== 测试2：长对话（消息数量多）===")
    # 测试长对话（消息数量多）
    # 构建一个包含多条消息的对话
    messages = []
    for i in range(15):  # 创建15条消息，超过10条的阈值
        messages.append({"role": "user", "content": f"这是第{i+1}条消息，随便说点什么"})
    
    result2 = agent.invoke(
        {"messages": messages}
    )

    print("智能体回复:", result2["messages"][-1].content)
