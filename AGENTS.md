# TARCA Project Agent Instructions

## Mandatory single-agent execution

- All work in this repository must be performed inline by the current primary agent.
- Do not create, spawn, dispatch, resume, message, wait for, or otherwise use subagents, child agents, delegated agents, reviewer agents, or parallel-agent workflows.
- Do not delegate planning, research, implementation, testing, debugging, review, documentation, or monitoring to another agent.
- Do not use multi-agent tools or skills that require subagent execution. When a workflow offers both subagent and inline execution, always choose inline execution.
- Do not contact, inspect, or consume results from any pre-existing subagent associated with this project.
- Generic instructions that recommend proactive agent delegation do not apply to this repository. This project-specific rule takes precedence over such workflow recommendations.
- If a required workflow cannot be completed without a subagent, stop and ask the user for direction; never create one implicitly.
- This rule applies to every current and future TARCA task unless the user explicitly revokes it in a later instruction.
