import asyncio
from tools.plugins.file_tool import FileEditArgs, FileEditTool

# 初始化工具
file_edit_tool = FileEditTool()

# 定义异步执行函数
async def write_file():
    # 构建参数：覆盖写入 work/test.py 文件
    args = FileEditArgs(
        file_path="./work/test.py",
        action="create",
        replacement="xxx"
    )
    ctx = {
            "todo_store": {}, 
            "session_id": 1, 
            "agent_id": 1, 
            "sandbox_read_dirs": ["./"],
            "sandbox_write_dirs": ["./work"]
        }
    # 执行文件写入
    await file_edit_tool.execute(ctx, args)

# 运行异步任务
if __name__ == "__main__":
    # 创建 work 目录（避免目录不存在报错）
    import os
    os.makedirs("work", exist_ok=True)
    
    # 执行异步函数
    asyncio.run(write_file())