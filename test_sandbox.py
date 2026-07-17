from factory.sandbox.security import check_code, UnsafeCodeError
from factory.agents import CoderAgent
from factory.llm import MockClient

code = CoderAgent(MockClient())._template("IsolationForest", 0.03)
print("安全通过" if check_code(code) else "不安全")

try:
    check_code("""
import os
os.system("rm -rf /")
""")
    print("漏网")
except UnsafeCodeError:
    print("拦截成功")