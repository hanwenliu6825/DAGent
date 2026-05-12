"""DAGent tool schemas (partial release).

Only the DAGent-specific tools (Eq.1 QueryDoc fields exposed as the `finish`
schema, plus the Orchestrator's `plan` schema and the `recall` deep-retrieval
tool) are included in this partial release. Baseline tool schemas
(`search_tool`, `branch_tool`) and the sequential `TOOL_PROMPT` template have
been removed and will be released after acceptance.
"""


def convert_tools_to_description(tools: list[dict]) -> str:
    ret = ''
    for i, tool in enumerate(tools):
        assert tool['type'] == 'function'
        fn = tool['function']
        if i > 0:
            ret += '\n'
        ret += f'---- BEGIN FUNCTION #{i + 1}: {fn["name"]} ----\n'
        ret += f'Description: {fn["description"]}\n'

        if 'parameters' in fn:
            ret += 'Parameters:\n'
            properties = fn['parameters'].get('properties', {})
            required_params = set(fn['parameters'].get('required', []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                is_required = param_name in required_params
                param_status = 'required' if is_required else 'optional'
                param_type = param_info.get('type', 'string')
                desc = param_info.get('description', 'No description provided')
                if 'enum' in param_info:
                    enum_values = ', '.join(f'`{v}`' for v in param_info['enum'])
                    desc += f'\nAllowed values: [{enum_values}]'
                ret += f'  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n'
        else:
            ret += 'No parameters are required for this function.\n'

        ret += f'---- END FUNCTION #{i + 1} ----\n'
    return ret


PARALLEL_TOOL_PROMPT = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""


def search_tool_dagent():
    """DAGent ReAct Executor tools: search, open_page, recall, finish.

    The `finish` schema's five required fields (answer / explanation / key_steps /
    confidence / uncertainties) directly correspond to the QueryDoc D_i =
    (alpha, expl, steps, conf, unc) tuple in Paper Equation 1.
    """
    search = {
        'type': 'function',
        'function': {
            "name": "search",
            "description": "Performs a search: supply a string 'query' and optional 'topk'. The tool retrieves the top 'topk' results (default 10) for the query, returning their docid, url, and document content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query string for the search."},
                    "topk": {"type": "integer", "description": "Return the top k pages."}
                },
                "required": ["query"]
            }
        }
    }
    open_page = {
        'type': 'function',
        'function': {
            'name': 'open_page',
            'description': "Open a page by docid or URL and return the complete content.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'docid': {'type': 'string', 'description': 'Document ID from search results.'},
                    'url': {'type': 'string', 'description': 'URL from search results.'},
                },
                'required': [],
            },
        },
    }
    # QueryDoc D_i (Paper Eq.1): structured per-node evidence.
    finish = {
        'type': 'function',
        'function': {
            'name': 'finish',
            'description': "Return the final result with structured output including answer, explanation, key steps, confidence, and uncertainties.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'answer': {'type': 'string', 'description': 'DIRECT answer only (alpha_i in Eq.1). Do NOT include explanations here.'},
                    'explanation': {'type': 'string', 'description': 'Brief explanation with [docid] citations (expl_i in Eq.1).'},
                    'key_steps': {'type': 'string', 'description': 'Key research steps taken (steps_i in Eq.1), separated by semicolons.'},
                    'confidence': {'type': 'string', 'description': 'Confidence score (conf_i in Eq.1) between 0% and 100%.'},
                    'uncertainties': {'type': 'string', 'description': 'Unresolved questions or single-source facts (unc_i in Eq.1), separated by semicolons.'},
                },
                'required': ['answer', 'explanation', 'key_steps', 'confidence', 'uncertainties'],
            },
        },
    }
    # RecallTool (Paper Eq.4): on-demand retrieval from a dependency's InteractionTranscript.
    recall = {
        'type': 'function',
        'function': {
            'name': 'recall',
            'description': "Deep retrieval from a previous task node's original transcript. Use when upstream context summary is insufficient.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'node_id': {'type': 'string', 'description': 'The ID of the dependency node to recall from (e.g., "t1", "t2"). Must be in dep_i.'},
                    'goal': {'type': 'string', 'description': 'What specific information you need (the goal argument in Eq.4).'},
                },
                'required': ['node_id', 'goal'],
            },
        },
    }
    return [search, open_page, recall, finish]


def plan_tool():
    """Orchestrator structured planning tool (Paper Eq.3 P_k = (S_k, V_k^new))."""
    plan = {
        'type': 'function',
        'function': {
            'name': 'plan',
            'description': "Output incremental planning decision: evaluate graph state and decide next batch of tasks (Paper Eq.3).",
            'parameters': {
                'type': 'object',
                'properties': {
                    'node_statuses': {
                        'type': 'object',
                        'description': 'State evaluation S_k for nodes showing "Awaiting Evaluation". Keys are node IDs (e.g., "t1"), values are: "Success" / "Uncertain" / "Not Found". Use empty {} for first iteration.',
                        'additionalProperties': {'type': 'string'}
                    },
                    'nodes': {
                        'type': 'array',
                        'description': 'Targeted expansion V_k^new: new task nodes to add to the graph.',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'id': {'type': 'string', 'description': 'Unique node ID (e.g., "t1", "t2"). Auto-generated if empty.'},
                                'description': {'type': 'string', 'description': 'Self-contained task description (delta_i).'},
                                'prompt': {'type': 'string', 'description': 'Detailed execution instructions (p_i).'},
                                'dependencies': {
                                    'type': 'array',
                                    'items': {'type': 'string'},
                                    'description': 'List of node IDs this task depends on (dep_i). Only QueryDocs from these nodes are passed by Selective Propagation.'
                                }
                            },
                            'required': ['description', 'prompt']
                        }
                    },
                },
                'required': ['node_statuses', 'nodes'],
            },
        },
    }
    return [plan]
