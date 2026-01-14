# ToolStrategy结构化输出示例（DeepSeek）
# 本示例演示如何使用ToolStrategy获取结构化输出

# 导入必要的库
from pydantic import BaseModel  # 用于定义数据模型
from langchain.agents import create_agent  # 用于创建智能体
from langchain.agents.structured_output import ToolStrategy  # 用于结构化输出
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
    return f"搜索结果：{query}"

# 定义结构化输出模型
# 使用Pydantic的BaseModel定义联系信息模型
class ContactInfo(BaseModel):
    """联系信息模型"""
    name: str  # 姓名
    email: str  # 邮箱
    phone: str  # 电话

print("结构化输出模型定义完成，开始创建模型和智能体...")

# 创建ChatDeepSeek模型实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

# 创建带结构化输出的智能体
# 使用ToolStrategy指定结构化输出策略
agent = create_agent(
    model=model,  # 模型实例
    tools=[search],  # 传入搜索工具
    response_format=ToolStrategy(ContactInfo)  # 指定结构化输出策略
)

print("智能体创建完成，开始测试结构化输出...")

# 测试结构化输出
if __name__ == "__main__":
    print("=== 测试：结构化输出 ===")
    # 调用智能体，请求提取联系信息
    result = agent.invoke({
        "messages": [{"role": "user", "content": "从以下内容提取联系信息：John Doe, john@example.com, (555) 123-4567"}]
    })
    
    # 打印结构化响应
    print("结构化响应:")
    print(result["structured_response"])
    print()
    
    # 访问结构化响应的字段
    print("=== 结构化响应字段访问 ===")
    print(f"姓名: {result["structured_response"].name}")
    print(f"邮箱: {result["structured_response"].email}")
    print(f"电话: {result["structured_response"].phone}")
    print()
    
    # 分析响应结构
    print("=== 响应结构分析 ===")
    print(f"响应类型: {type(result)}")
    print(f"响应包含的键: {list(result.keys())}")
    print(f"结构化响应类型: {type(result["structured_response"])}")
