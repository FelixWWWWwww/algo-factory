"""
factory/llm/prompt_manager.py
管理和渲染 Jinja2 提示词模板
"""

import logging
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


class PromptManager:
    """提示词管理器"""

    def __init__(self, template_dir: str = "factory/llm/prompts"):
        self.template_dir = Path(template_dir)
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        logger.info(f"[PromptManager] 初始化，模板目录: {self.template_dir}")

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        try:
            prompt_file = self.template_dir / "system_prompt.txt"
            return prompt_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("[PromptManager] system_prompt.txt 未找到")
            return ""

    def render_template(self, template_name: str, **context) -> str:
        """
        渲染 Jinja2 模板

        Args:
            template_name: 模板文件名（如 "model_selection_explanation.jinja2"）
            **context: 模板变量

        Returns:
            渲染后的提示词
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except TemplateNotFound:
            logger.error(f"[PromptManager] 模板未找到: {template_name}")
            raise

    def get_planner_prompt(
            self,
            user_query: str,
            n_samples: int,
            n_features: int,
            anomaly_ratio: float,
            has_labels: bool,
            retrieved_context: str
    ) -> str:
        """为 Planner Agent 生成 prompt"""
        return self.render_template(
            "planner_prompt.jinja2",
            user_query=user_query,
            n_samples=n_samples,
            n_features=n_features,
            anomaly_ratio=anomaly_ratio,
            has_labels=has_labels,
            retrieved_context=retrieved_context
        )

    def get_model_selection_prompt(
            self,
            results: list,
            best_algorithm: str,
            second_algorithm: str,
            n_samples: int,
            n_features: int,
            anomaly_ratio: float,
            has_labels: bool
    ) -> str:
        """为模型选择生成解释 prompt"""
        return self.render_template(
            "model_selection_explanation.jinja2",
            results=results,
            best_algorithm=best_algorithm,
            second_algorithm=second_algorithm,
            n_samples=n_samples,
            n_features=n_features,
            anomaly_ratio=anomaly_ratio,
            has_labels=has_labels
        )


# 全局实例
_prompt_manager = None


def get_prompt_manager() -> PromptManager:
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
