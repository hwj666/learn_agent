import json
import re
from typing import List, Dict
from agent_core import AgentCore
class PlanAgent:
    def __init__(self, agent_core: AgentCore, max_steps: int = 10):
        self.executor = agent_core  # 组合 AgentCore
        self.max_steps = max_steps
        self.completed_tasks = []    # 存储已完成任务的简报
        self.current_plan = []       # 当前待执行的任务队列
        self.final_answer = ""

    def _extract_json(self, text: str):
        """鲁棒的 JSON 提取逻辑"""
        try:
            # 优先匹配 Markdown 代码块中的 JSON
            json_block = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            content = json_block.group(1) if json_block else text
            # 寻找数组边界
            match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            return None
        except Exception:
            return None

    def _planner(self, goal: str, last_result: str = None) -> bool:
        """决策层：判断是继续规划新步骤，还是已经完成"""
        status_context = ""
        if self.completed_tasks:
            status_context = "\n".join([f"- 步骤{i+1}: {t}" for i, t in enumerate(self.completed_tasks)])
        
        prompt = f"""
        【总目标】: {goal}
        【当前进展】: {status_context}
        【最新一步结果】: {last_result if last_result else "尚未开始"}

        请作为架构师进行决策：
        1. 如果目标已达成，或者无法再取得进展，请回复：["FINISH"]
        2. 如果需要继续，请输出接下来 1-2步 具体的执行计划（JSON 格式）。
           格式：[ {{"task": "任务描述", "reason": "理由"}} ]
        
        注意：只需输出 JSON 数组或 ["FINISH"]，不要任何解释。
        """
        
        # 调用核心进行思考决策
        response = self.executor.run(prompt)
        
        if "FINISH" in response:
            return False
            
        new_steps = self._extract_json(response)
        if new_steps:
            self.current_plan = new_steps
            return True
        return False

    def run(self, goal: str):
        print(f"\n\033[1;32m🚀 开始执行任务: {goal}\033[0m")
        
        step_count = 0
        last_step_result = None

        # 进入“规划-执行”循环
        while step_count < self.max_steps:
            # 1. 动态规划/重规划
            has_next = self._planner(goal, last_step_result)
            
            if not has_next:
                print("\033[1;32m✅ 规划器确认任务结束。\033[0m")
                break
            
            # 2. 执行当前计划中的任务
            for task in self.current_plan:
                step_count += 1
                print(f"\n\033[1;34m[步骤 {step_count}] 正在执行: {task['task']}\033[0m")
                
                # 构造执行指令：注入全局目标和之前的精简记忆
                execute_prompt = (
                    f"【总目标】: {goal}\n"
                    f"【历史摘要】: {' | '.join(self.completed_tasks[-3:])}\n" # 只给最近3步摘要，防冗余
                    f"【当前任务】: {task['task']}\n"
                    "请直接利用工具完成该任务并给出结论。"
                )
                
                # 执行前清空 Core 的琐碎历史，保持专注
                self.executor.history = [] 
                last_step_result = self.executor.run(execute_prompt)
                
                # 记录结果（此处可以加一步让模型自动对结果做摘要）
                self.completed_tasks.append(f"任务: {task['task']} -> 结果: {last_step_result[:100]}...")
            
            # 清空当前执行完的短计划，回到 while 循环触发下一次 planner 评估
            self.current_plan = []

        # 3. 最终汇总
        print("\n\033[1;36m🏁 正在生成最终报告...\033[0m")
        summary_prompt = f"基于以下执行记录，给出最终答复：\n目标：{goal}\n过程：{self.completed_tasks}"
        self.final_answer = self.executor.run(summary_prompt)
        return self.final_answer


if __name__ == "__main__":
    from registry_tool import get_all_tools
    import code_tools
    # 初始化 Agent

    system = """
    【核心执行规则】
1. 你必须先执行检查任务，若发现任何错误/异常/问题：
   ✅ 必须立即调用【修复工具】，绝对不能直接回答任务完成
   ✅ 禁止省略工具调用，禁止跳过修复步骤
2. 只有在调用工具并确认修复成功后，才能判定任务完成
3. 未调用修复工具前，永远不能输出「任务已修复/已完成」
    """
    agent = AgentCore(model="qwen-max",system_prompt=system)

    # 🔥 自动注册所有调试工具
    agent.bind_tools(get_all_tools())
    planner = PlanAgent(agent)
    # ===================== 测试指令（直接改这里）=====================
    planner.run("帮我分析当前代码结构")