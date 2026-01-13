# ReAct 循环中的工具使用示例（DeepSeek）
# 本示例演示智能体如何在ReAct循环中使用工具

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
# 搜索工具
@tool
def search_products(query: str) -> str:
    """
    搜索产品信息

    参数：
    - query: str类型，产品搜索关键词

    返回值：
    - str类型，搜索结果
    """
    query_lower = query.lower()

    # 根据不同的查询返回不同的结果
    if any(keyword in query_lower for keyword in ["无线耳机", "耳机", "蓝牙耳机"]):
        return "搜索到以下热门无线耳机：\n1. WH-1000XM5（索尼）\n2. AirPods Pro 2（苹果）\n3. Bose QuietComfort Ultra\n4. Sony WF-1000XM5\n5. Sennheiser Momentum True Wireless 4"
    elif any(keyword in query_lower for keyword in ["airpods", "苹果耳机"]):
        return "搜索到苹果耳机：AirPods Pro 2（第二代），具备主动降噪功能，续航6小时"
    elif any(keyword in query_lower for keyword in ["索尼", "sony"]):
        return "搜索到索尼耳机：WH-1000XM5（头戴式），WF-1000XM5（真无线），均具备行业领先的降噪技术"
    elif "最受欢迎" in query_lower or "热门" in query_lower:
        return "当前最受欢迎的无线耳机排行榜：\n1. AirPods Pro 2（苹果）\n2. WH-1000XM5（索尼）\n3. Bose QuietComfort Ultra\n4. Galaxy Buds2 Pro（三星）\n5. Sony WF-1000XM5"
    else:
        return f"搜索结果：{query} - 找到3个相关产品，请提供更具体的搜索词"


# 库存检查工具
@tool
def check_inventory(product_id: str) -> str:
    """
    检查产品库存

    参数：
    - product_id: str类型，产品ID

    返回值：
    - str类型，库存信息
    """
    # 模拟库存信息
    inventory_data = {
        "WH-1000XM5": "库存 10 件",
        "AirPods Pro 2": "库存 5 件",
        "Bose QuietComfort Ultra": "库存 3 件",
        "Sony WF-1000XM5": "库存 8 件",
        "Sennheiser Momentum True Wireless 4": "库存 2 件"
    }
    return f"产品 {product_id}：{inventory_data.get(product_id, '库存 0 件')}"


print("工具定义完成，开始创建模型和智能体...")

# 创建ChatDeepSeek模型实例
model = ChatDeepSeek(
    model="deepseek-chat",  # 模型名称
    api_key=api_key  # 传入API密钥
)

# 创建智能体，传入模型和工具
agent = create_agent(
    model=model,  # 模型实例
    tools=[search_products, check_inventory]  # 传入工具列表
)

print("智能体创建完成，开始测试ReAct循环...")

# 测试智能体的逻辑思考能力
if __name__ == "__main__":
    print("=== 测试1：完整查询流程 ===")
    print("用户：找出当前最受欢迎的无线耳机并检查其库存")
    print()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "找出当前最受欢迎的无线耳机并检查其库存"}]}
    )

    print("智能体回复:", result["messages"][-1].content)
    print()

    print("=== 测试2：直接查询特定产品 ===")
    print("用户：检查WH-1000XM5的库存")
    print()

    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "检查WH-1000XM5的库存"}]}
    )

    print("智能体回复:", result2["messages"][-1].content)