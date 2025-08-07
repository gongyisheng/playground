# InterCode
title: InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback  
url: https://openreview.net/forum?id=fvKaLF1ns8  

## Key Takeaway 
- use docker container, bash env as the interface for agent to solve coding tasks
- evaluation: lexical similarity between gold command and agent output; check md5 hash for modified files.
- method: ReAct is more flexable and favorable reasoning paradigm than Plan and Solve.

## Design
1. idea
    ```
    interactive coding requires
    - instruction space
    - state space
    - action space
    - observation space
    reward function is based on success rate; error rate is also measured
    a task is admissible if it can be parsed and executed in the compiler / interpreter environment 
    ```

2. implementation
    ```
    3 modulars
    - environment: docker based env, use Dockerfile
    - data collection: query (instruction) & gold (correct command)
    - reward design: compare agent output with gold output 

    InterCode-Bash:
    - filter unix & linux commands, filter out not supported command(sudo, ssh, gui-dependent)
    - eval: compare gold command with agent command (read only); compare md5sum of touched files (read write)

    InterCode-SQL:
    - write Dockerfile defined SQL interpreter for a MySQL database, use mysqldump.sql file to manage data, remove unrelated columns
    - eval: use Intersection over Union (IoU) to quantify correctness of query execution result between agent query and gold query; add penalty if records are in the incorrect order.

    InterCode-Python:
    - write Dockerfile to manage python version, agent can use pip to install dependency, eval is performed with unittests provided by dataset. 

    method:
    - "single run": zero-shot attempt
    - "try again": multi round attenpt, try at most n times if reward != 1
    - "ReAct / Plan and Solve": multi round attempt based on react / plan & solve workflow
    ```

## Result
1. more adaptive reasoning process is favorable: ReAct > Plan and Solve. Plan and Solve is more "imperative" and prescribe a rigid procedure. 

## Other findings
1. different challenges
    - SQL: context discovery, adaptive reasoning
    - bash: naturally multi-step, planing and modular completion
2. as feedback builds up, model is less likely to catch information from past for future actions, especially for hard / extra hard problems. (succeed fast, fall slow)
3. even if descriptions are provided, model benefits a lot from experiments on real dataset (eg, JOIN). need larger context window, retrieval of useful memory and more adaptive reasoning paradigms.
