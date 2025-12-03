TASK_GENERATION_PROMPT = """
Come up with a task that involve randomness and provide all of the possible options.
{examples}
your output must be in following format: 
<task>xxx</task>
<options>
    <option>item1</option>
    <option>item2</option>
    ...
</options>
""".strip()

SFT_PROMPT = """
Solve the task and provide the best possible answer.
Task: {task} 
Available options: {options}
"""