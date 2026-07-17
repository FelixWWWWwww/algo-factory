# factory/sandbox/security.py
import ast


class UnsafeCodeError(Exception):
    """检测到危险代码时抛出。"""


_BLOCKED_FUNCS = {"eval", "exec", "compile", "__import__"}
_BLOCKED_ATTR = {"system", "popen", "rmtree", "remove", "unlink", "rmdir", "kill", "fork"}
_BLOCKED_IMPORTS = {"subprocess", "socket", "shutil"}


def check_code(code: str) -> bool:
    """AST 遍历黑名单检查，安全返回 True，否则抛 UnsafeCodeError。"""
    tree = ast.parse(code)  # 顺带做语法检查
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in _BLOCKED_IMPORTS:
                    raise UnsafeCodeError(f"禁止导入模块: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _BLOCKED_IMPORTS:
                raise UnsafeCodeError(f"禁止导入模块: {node.module}")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _BLOCKED_FUNCS:
                raise UnsafeCodeError(f"禁止调用: {f.id}()")
            if isinstance(f, ast.Attribute) and f.attr in _BLOCKED_ATTR:
                raise UnsafeCodeError(f"禁止调用危险方法: .{f.attr}()")
    return True