# test_llm.py
import os
import sys
from dotenv import load_dotenv
from pydantic import BaseModel

# 导入我们写的代码
from factory.llm import MockClient, OpenAIClient, StructuredOutput

# 加载环境变量（如果需要真实 API）
load_dotenv()


# 定义一个 Pydantic 模型（测试数据结构）
class TaskCard(BaseModel):
    task_type: str
    target: str
    constraints: list
    metrics: list
    data_hint: str


def test_mock_client():
    """测试 Mock 客户端"""
    print("=" * 50)
    print("🧪 测试 Mock 客户端")
    print("=" * 50)

    # 创建 Mock 客户端
    mock_client = MockClient()

    # 模拟 Interpreter Agent 的调用
    messages = [
        {
            "role": "user",
            "content": "请分析任务：构建交易异常检测模型"
        }
    ]

    response = mock_client.chat(messages)
    print(f"✅ Mock 响应:")
    print(response["message"])
    print()


def test_structured_output():
    """测试结构化输出处理器"""
    print("=" * 50)
    print("🧪 测试结构化输出处理器")
    print("=" * 50)

    # 创建 Mock 客户端
    mock_client = MockClient()

    # 创建结构化输出处理器
    structured = StructuredOutput(mock_client)

    # 测试调用
    try:
        prompt = "分析任务：构建交易异常检测模型"
        result = structured.call_structured(
            prompt=prompt,
            schema=TaskCard
        )

        print(f"✅ 解析成功!")
        print(f"   Task Type: {result.task_type}")
        print(f"   Target: {result.target}")
        print(f"   Metrics: {result.metrics}")
    except Exception as e:
        print(f"❌ 解析失败: {e}")

    print()


def test_openai_client():
    """测试 OpenAI 客户端（需要 API Key）"""
    print("=" * 50)
    print("🧪 测试 OpenAI 客户端")
    print("=" * 50)

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")

    if not api_key or not base_url:
        print("⚠️  未找到完整的 LLM 环境变量，跳过此测试")
        print("   需要设置：DEEPSEEK_API_KEY / OPENAI_API_KEY 以及 OPENAI_BASE_URL / LLM_BASE_URL")
        return

    try:
        client = OpenAIClient(api_key=api_key, base_url=base_url)

        messages = [{"role": "user", "content": "Say hello"}]
        response = client.chat(messages)

        print(f"✅ OpenAI 响应:")
        print(response["message"])
        print(f"   Tokens: {response['usage']}")
    except Exception as e:
        print(f"❌ 调用失败: {e}")

    print()


if __name__ == "__main__":
    print("\n🚀 开始测试 LLM 抽象层\n")

    # 运行测试
    test_mock_client()
    test_structured_output()
    test_openai_client()

    print("=" * 50)
    print("✅ 所有测试完成！")
    print("=" * 50)
