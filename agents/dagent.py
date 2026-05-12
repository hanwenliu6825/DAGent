"""
DAGent: DAG-Based Multi-Agent Framework for Deep Research

Core components:
- TaskDAG: Directed acyclic graph of sub-tasks
- TaskNode: Node with two-level attributes (QueryDoc, InteractionTranscript)
- Orchestrator: Evaluate-then-Grow incremental planner
- RecallTool: On-demand retrieval from InteractionTranscript
"""

import os
import re
import json
import copy
import time
import asyncio
import random
from uuid import uuid4
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union

from verl import DataProto
from .utils import Agent, select_env, TaskContext, run_action, AgentLoopOutput, AgentLoopMetrics, ContextOverflowError, _chat_template_kwargs
from .prompts import create_chat

try:
    import json_repair
except ImportError:
    json_repair = None


# ============================================================================
# InteractionTranscript - full execution record (node attribute)
# ============================================================================

@dataclass
class InteractionTranscript:
    """Full execution record - preserves the ReAct trace for RecallTool deep retrieval."""
    raw_text: str = ""
    turns: int = 0
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'raw_text': self.raw_text,
            'turns': self.turns,
            'failure_reason': self.failure_reason
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'InteractionTranscript':
        return cls(
            raw_text=data.get('raw_text', ''),
            turns=data.get('turns', 0),
            failure_reason=data.get('failure_reason')
        )


# ============================================================================
# QueryDoc - structured summary (node attribute)
# ============================================================================

@dataclass
class QueryDoc:
    """Structured summary - used for selective propagation and Orchestrator state evaluation."""
    answer: str = ""
    explanation: str = ""
    key_steps: str = ""
    confidence: str = ""
    uncertainties: str = ""

    def to_summary(self, node_id: str = "", description: str = "", status: str = "") -> str:
        """Generate the summary text used for selective propagation."""
        lines = []
        if node_id:
            lines.append(f"[Node {node_id}] {description}")
        if status:
            lines.append(f"Status: {status}")
        if self.answer:
            lines.append(f"Answer: {self.answer}")
        if self.explanation:
            lines.append(f"Explanation: {self.explanation}")
        if self.key_steps:
            lines.append(f"Key Steps: {self.key_steps}")
        if self.confidence:
            lines.append(f"Confidence: {self.confidence}")
        if self.uncertainties:
            lines.append(f"Uncertainties: {self.uncertainties}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            'answer': self.answer,
            'explanation': self.explanation,
            'key_steps': self.key_steps,
            'confidence': self.confidence,
            'uncertainties': self.uncertainties
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'QueryDoc':
        return cls(
            answer=data.get('answer', ''),
            explanation=data.get('explanation', ''),
            key_steps=data.get('key_steps', ''),
            confidence=data.get('confidence', ''),
            uncertainties=data.get('uncertainties', '')
        )

    @classmethod
    def from_finish_output(cls, finish_output: dict) -> 'QueryDoc':
        """Create a QueryDoc from the ReAct executor's finish tool output."""
        return cls(
            answer=finish_output.get('answer', ''),
            explanation=finish_output.get('explanation', ''),
            key_steps=finish_output.get('key_steps', ''),
            confidence=finish_output.get('confidence', '0%'),
            uncertainties=finish_output.get('uncertainties', '')
        )


# ============================================================================
# TaskNode - DAG node
# ============================================================================

@dataclass
class TaskNode:
    """
    DAG node - carries two attribute layers:
    - QueryDoc: structured summary (used for selective propagation and state evaluation)
    - InteractionTranscript: full execution record (used for RecallTool deep retrieval)
    """
    # Graph structure fields
    id: str
    description: str
    prompt: str
    dependencies: List[str] = field(default_factory=list)
    status: str = ""             # "" | "Success" | "Uncertain" | "Not Found" | "Failed"
    depth: int = 0               # Node depth (longest path from a root)

    # Node attributes
    query_doc: QueryDoc = field(default_factory=QueryDoc)
    transcript: InteractionTranscript = field(default_factory=InteractionTranscript)

    # === QueryDoc field convenience accessors ===
    @property
    def answer(self) -> str:
        return self.query_doc.answer

    @answer.setter
    def answer(self, value: str):
        self.query_doc.answer = value

    @property
    def explanation(self) -> str:
        return self.query_doc.explanation

    @explanation.setter
    def explanation(self, value: str):
        self.query_doc.explanation = value

    @property
    def key_steps(self) -> str:
        return self.query_doc.key_steps

    @key_steps.setter
    def key_steps(self, value: str):
        self.query_doc.key_steps = value

    @property
    def confidence(self) -> str:
        return self.query_doc.confidence

    @confidence.setter
    def confidence(self, value: str):
        self.query_doc.confidence = value

    @property
    def uncertainties(self) -> str:
        return self.query_doc.uncertainties

    @uncertainties.setter
    def uncertainties(self, value: str):
        self.query_doc.uncertainties = value

    def get_transcript_text(self) -> str:
        return self.transcript.raw_text

    def set_transcript_text(self, text: str, turns: int = 0, failure_reason: str = None):
        self.transcript = InteractionTranscript(
            raw_text=text, turns=turns, failure_reason=failure_reason
        )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'description': self.description,
            'prompt': self.prompt,
            'dependencies': self.dependencies,
            'status': self.status,
            'depth': self.depth,
            'query_doc': self.query_doc.to_dict(),
            'transcript': self.transcript.to_dict(),
            'answer': self.query_doc.answer,
            'explanation': self.query_doc.explanation,
            'key_steps': self.query_doc.key_steps,
            'confidence': self.query_doc.confidence,
            'uncertainties': self.query_doc.uncertainties
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskNode':
        query_doc = QueryDoc.from_dict(data['query_doc']) if isinstance(data.get('query_doc'), dict) else QueryDoc()
        transcript = InteractionTranscript.from_dict(data['transcript']) if isinstance(data.get('transcript'), dict) else InteractionTranscript()
        return cls(
            id=data['id'],
            description=data.get('description', ''),
            prompt=data.get('prompt', ''),
            dependencies=data.get('dependencies', []),
            status=data.get('status', ''),
            depth=data.get('depth', 0),
            query_doc=query_doc,
            transcript=transcript
        )

    def get_query_doc_summary(self) -> str:
        """Return the QueryDoc summary (used for selective propagation)."""
        return self.query_doc.to_summary(
            node_id=self.id, description=self.description, status=self.status
        )


# ============================================================================
# TaskNodeSpec - Orchestrator output node specification
# ============================================================================

@dataclass
class TaskNodeSpec:
    """Orchestrator output node specification."""
    id: str
    description: str
    prompt: str
    dependencies: List[str]

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskNodeSpec':
        return cls(
            id=data.get('id', ''),
            description=data.get('description', ''),
            prompt=data.get('prompt', ''),
            dependencies=data.get('dependencies', [])
        )


# ============================================================================
# PlanOutput - Orchestrator planning output
# ============================================================================

@dataclass
class PlanOutput:
    """Orchestrator planning output."""
    node_statuses: Dict[str, str]    # {node_id: status}
    nodes: List[TaskNodeSpec]        # newly added nodes for this iteration


# ============================================================================
# TaskDAG - DAG container class
# ============================================================================

class TaskDAG:
    """
    Task DAG container class.

    Core responsibilities:
    1. Manage TaskNode nodes and dependency edges.
    2. Selective Propagation: get_upstream_context() returns only direct-dependency QueryDocs.
    3. Evaluate-then-Grow: nodes become executable as soon as they are added.
    """

    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self._node_counter = 0

    def add_node(self, node: TaskNode) -> str:
        """Add a node and automatically maintain reverse dependency edges."""
        if not node.id:
            self._node_counter += 1
            node.id = f"t{self._node_counter}"
        self.nodes[node.id] = node
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                self.edges.append((dep_id, node.id))
        return node.id

    def add_node_from_spec(self, spec: dict) -> str:
        node = TaskNode(
            id=spec.get('id', ''),
            description=spec.get('description', ''),
            prompt=spec.get('prompt', ''),
            dependencies=spec.get('dependencies', [])
        )
        return self.add_node(node)

    def get_node(self, node_id: str) -> Optional[TaskNode]:
        return self.nodes.get(node_id)

    def get_all_nodes(self) -> List[TaskNode]:
        return list(self.nodes.values())

    def update_node(self, node_id: str, **kwargs) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            return False
        for key, value in kwargs.items():
            if key == 'query_doc' and isinstance(value, QueryDoc):
                node.query_doc = value
            elif key == 'transcript' and isinstance(value, InteractionTranscript):
                node.transcript = value
            elif key == 'transcript' and isinstance(value, str):
                node.transcript.raw_text = value
            elif hasattr(node, key):
                setattr(node, key, value)
        return True

    def get_upstream_context(self, node_id: str) -> str:
        """
        Selective Propagation: returns only the QueryDocs of direct-dependency parents.
        per-node context bound: O(k · |D|)
        """
        node = self.nodes.get(node_id)
        if not node:
            return ""
        if not node.dependencies:
            return "[No upstream dependencies]"
        context_parts = []
        for dep_id in node.dependencies:
            dep_node = self.nodes.get(dep_id)
            if dep_node:
                context_parts.append(dep_node.get_query_doc_summary())
        if not context_parts:
            return "[Upstream dependencies not found in graph]"
        return "\n\n---\n\n".join(context_parts)

    def to_prompt_format(self) -> str:
        """Convert the graph into the Orchestrator-readable state format."""
        if not self.nodes:
            return "[Empty graph - initial planning]"
        lines = [f"Total nodes: {len(self.nodes)}", "", "=== Executed Nodes ==="]
        for node in self.nodes.values():
            deps_str = f" (deps: {', '.join(node.dependencies)})" if node.dependencies else ""
            lines.append(f"[{node.id}]: {node.description}")
            lines.append(f"    Status: {node.status or 'Awaiting Evaluation'}{deps_str}")
            if node.answer:
                lines.append(f"    Answer: {node.answer}")
            if node.explanation:
                lines.append(f"    Explanation: {node.explanation}")
            if node.key_steps:
                lines.append(f"    Key Steps: {node.key_steps}")
            if node.confidence:
                lines.append(f"    Confidence: {node.confidence}")
            if node.uncertainties:
                lines.append(f"    Uncertainties: {node.uncertainties}")
            lines.append("")
        return "\n".join(lines)

    def to_delta_update(self, new_node_ids: List[str], status_changes: Dict[str, str] = None) -> str:
        """Generate delta update text for newly executed nodes and status changes."""
        lines = []
        if status_changes:
            lines.append("=== Status Changes ===")
            for node_id, status in status_changes.items():
                lines.append(f"[{node_id}]: {status}")
            lines.append("")

        new_nodes = [self.nodes[nid] for nid in new_node_ids if nid in self.nodes]
        if new_nodes:
            lines.append("=== New Execution Results ===")
            for node in new_nodes:
                deps_str = f" (deps: {', '.join(node.dependencies)})" if node.dependencies else ""
                lines.append(f"[{node.id}]: {node.description}")
                lines.append(f"    Status: {node.status or 'Awaiting Evaluation'}{deps_str}")
                if node.answer:
                    lines.append(f"    Answer: {node.answer}")
                if node.explanation:
                    lines.append(f"    Explanation: {node.explanation}")
                if node.key_steps:
                    lines.append(f"    Key Steps: {node.key_steps}")
                if node.confidence:
                    lines.append(f"    Confidence: {node.confidence}")
                if node.uncertainties:
                    lines.append(f"    Uncertainties: {node.uncertainties}")
                lines.append("")

        if not lines:
            return "[No new results]"
        return "\n".join(lines)

    def to_compact_ledger(self) -> str:
        """Generate global status ledger for Orchestrator context.

        Sections:
        - Statuses: every node id with its full-name status
        - Refine Budget: attempts used per root line of inquiry, with explicit
          warnings when budget is reached or about to be reached
        - Frontier: nodes awaiting evaluation and unresolved (Uncertain / Not Found)
        """
        if not self.nodes:
            return "[Empty graph]"
        lines = []

        # --- Statuses (full names, not abbreviations) ---
        lines.append("### Statuses")
        for node in self.nodes.values():
            status = node.status or 'Awaiting Evaluation'
            lines.append(f"{node.id}: {status}")

        # --- Refine Budget (count attempts per root line of inquiry) ---
        # An attempt is the original search node + each <root>_refine1/_refine2 child.
        # Budget = 3 attempts per line (original + refine1 + refine2).
        root_attempts = {}
        for node in self.nodes.values():
            if '_refine' in node.id:
                root_id = node.id.split('_refine')[0]
                root_attempts[root_id] = root_attempts.get(root_id, 1) + 1
            elif not node.id.startswith('t_'):
                if node.id not in root_attempts:
                    root_attempts[node.id] = 1
        # Only print this section when at least one line has been refined.
        if any(v > 1 for v in root_attempts.values()):
            lines.append("")
            lines.append("### Refine Budget (3 attempts max per line of inquiry: original + refine1 + refine2)")
            for root_id, used in root_attempts.items():
                if used <= 1:
                    continue
                exhausted = " [BUDGET REACHED]" if used >= 3 else ""
                lines.append(f"{root_id}: {used}/3 attempts used{exhausted}")

        # --- Frontier: awaiting evaluation and unresolved ---
        awaiting = [n.id for n in self.nodes.values() if not n.status]
        unresolved = [n.id for n in self.nodes.values() if n.status in ('Uncertain', 'Not Found')]
        if awaiting or unresolved:
            lines.append("")
            lines.append("### Frontier")
            if awaiting:
                lines.append(f"Awaiting Evaluation: {', '.join(awaiting)}")
            if unresolved:
                lines.append(
                    f"Unresolved (need refine if budget remains): "
                    f"{', '.join(unresolved)}"
                )

        return "\n".join(lines)

    def has_answer_node(self) -> bool:
        return any(node.id.endswith("_answer") for node in self.nodes.values())

    def get_answer_node(self) -> Optional[TaskNode]:
        answer_nodes = [node for node in self.nodes.values() if node.id.endswith("_answer")]
        return answer_nodes[-1] if answer_nodes else None

    def get_final_answer(self) -> Optional[str]:
        answer_node = self.get_answer_node()
        if answer_node and answer_node.answer:
            return answer_node.answer
        nodes_with_answer = [n for n in self.nodes.values() if n.answer]
        if nodes_with_answer:
            return nodes_with_answer[-1].answer
        return None

    def to_dict(self) -> dict:
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': self.edges
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskDAG':
        graph = cls()
        for node_id, node_data in data.get('nodes', {}).items():
            node = TaskNode.from_dict(node_data)
            graph.nodes[node_id] = node
        graph.edges = data.get('edges', [])
        return graph

    def compute_depths(self) -> int:
        """Update each node's depth (longest path from a root via topological traversal + DP)."""
        if not self.nodes:
            return 0
        from collections import deque
        for node in self.nodes.values():
            node.depth = 0
        in_degree = {nid: 0 for nid in self.nodes}
        children = {nid: [] for nid in self.nodes}
        for (src, dst) in self.edges:
            if dst in in_degree:
                in_degree[dst] += 1
            if src in children:
                children[src].append(dst)
        queue = deque()
        for nid, deg in in_degree.items():
            if deg == 0:
                self.nodes[nid].depth = 0
                queue.append(nid)
        max_depth = 0
        processed = 0
        while queue:
            nid = queue.popleft()
            current_depth = self.nodes[nid].depth
            max_depth = max(max_depth, current_depth)
            processed += 1
            for child_id in children[nid]:
                self.nodes[child_id].depth = max(
                    self.nodes[child_id].depth, current_depth + 1
                )
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)
        if processed != len(self.nodes):
            raise ValueError(
                f"TaskDAG is not a DAG: processed {processed}/{len(self.nodes)} nodes. "
                f"Graph may contain cycles or disconnected components without roots."
            )
        return max_depth

    def get_transitive_dependencies(self, node_id: str) -> set:
        """Return all transitive dependencies of node_id (inclusive of node_id itself)."""
        result = {node_id}
        queue = list(self.get_node(node_id).dependencies or [])
        while queue:
            dep = queue.pop()
            if dep not in result:
                result.add(dep)
                node = self.get_node(dep)
                if node and node.dependencies:
                    queue.extend(node.dependencies)
        return result

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"TaskDAG(nodes={len(self.nodes)}, edges={len(self.edges)})"


# ============================================================================
# RecallTool - deep retrieval from InteractionTranscript
# ============================================================================

class RecallTool:
    """Deep retrieval from a historical node's InteractionTranscript in the TaskDAG."""

    def __init__(self, graph: TaskDAG, llm_client, tokenizer):
        self.graph = graph
        self.llm_client = llm_client
        self.tokenizer = tokenizer

    def update_graph(self, graph: TaskDAG):
        self.graph = graph

    async def call(self, node_id: str, goal: str) -> str:
        node = self.graph.get_node(node_id)
        if not node:
            available_ids = list(self.graph.nodes.keys())
            return f"[Recall Error] Node with id '{node_id}' not found. Available nodes: {available_ids}"

        transcript_text = node.transcript.raw_text
        if not transcript_text:
            return f"[Recall Error] No transcript available for node '{node_id}'. The node may not have been executed yet."

        from .prompts import RECALL_EXTRACTION_PROMPT
        prompt = RECALL_EXTRACTION_PROMPT.format(goal=goal, transcript=transcript_text)
        messages = [{'role': 'user', 'content': prompt}]

        try:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **_chat_template_kwargs())
            config = self.llm_client.config
            max_len = len(input_ids) + config.response_length
            response = await self.llm_client.create_completion(
                input_ids=input_ids, messages=messages, max_len=max_len
            )
            if response:
                return f"[Recall from Node {node_id}]\n{response['choices'][0]['message']['content']}"
        except Exception:
            pass

        return self._simple_extract(node, goal)

    def _simple_extract(self, node: TaskNode, goal: str) -> str:
        keywords = goal.lower().split()
        lines = node.transcript.raw_text.split('\n')
        relevant = [line for line in lines if any(kw in line.lower() for kw in keywords)]
        if relevant:
            return f"[Recall from Node {node.id}]\nRelevant excerpts:\n" + "\n".join(relevant[:20])
        return f"[Recall from Node {node.id}]\nNo directly relevant content found for: {goal}"


# ============================================================================
# Orchestrator - Evaluate-then-Grow incremental planner
# ============================================================================

@dataclass
class PlanResult:
    """Typed result from Orchestrator.plan() for differentiated retry handling."""
    status: str  # "ok" | "parse_error" | "context_exhausted" | "llm_no_response" | "llm_exception"
    plan_output: PlanOutput = field(default_factory=lambda: PlanOutput(node_statuses={}, nodes=[]))
    raw_text: str = ""
    diagnostics: str = ""


def _normalize_quotes(text: str) -> str:
    """Convert single-quoted pseudo-JSON to double-quoted JSON."""
    if "'" in text and '"' not in text:
        return text.replace("'", '"')
    result = re.sub(r"(?<=[{,\s])'([^']+)'\s*:", r'"\1":', text)
    result = re.sub(r":\s*'([^']*)'", r': "\1"', result)
    return result


def _extract_json_block(text: str) -> Optional[str]:
    """Extract the last valid JSON block from text.

    Strategy: block location is lenient, structure validation is strict.
    1. Prefer last fenced ```json ... ``` block
    2. Fallback: backward brace-balanced scan
    """
    # 1. Fenced block (last one)
    fenced = list(re.finditer(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL))
    if fenced:
        return fenced[-1].group(1).strip()

    # 2. Backward brace-balanced scan: find last top-level {...}
    depth = 0
    end = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] == '}':
            if depth == 0:
                end = i
            depth += 1
        elif text[i] == '{':
            depth -= 1
            if depth == 0 and end != -1:
                return text[i:end + 1]

    return None


def _parse_plan_json(text: str) -> Optional[dict]:
    """Parse JSON with fallback chain (json.loads → json_repair → quote normalization)."""
    if not text:
        return None

    # 1. Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. json_repair
    if json_repair:
        try:
            parsed = json_repair.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 3. Quote normalization
    normalized = _normalize_quotes(text)
    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _validate_plan_schema(parsed: dict) -> bool:
    """Validate that parsed dict has required top-level plan structure.
    Both node_statuses and nodes must be present (can be empty {} / [])."""
    if not isinstance(parsed, dict):
        return False
    if 'node_statuses' not in parsed or 'nodes' not in parsed:
        return False
    if not isinstance(parsed['node_statuses'], dict):
        return False
    if not isinstance(parsed['nodes'], list):
        return False
    return True


def _normalize_search_text(text: str) -> str:
    """Normalize a description/prompt string for duplicate detection.
    Lowercase, collapse whitespace, strip surrounding spaces."""
    if not text:
        return ""
    return " ".join(text.split()).lower()


_REFINE_ID_RE = re.compile(r"^(.+?)_refine(\d+)$")


# Violation type taxonomy for C3 (Structural Compliance Penalty) metrics.
# Order-sensitive: R_REFINE must be checked BEFORE DEP because refine
# parent-missing errors contain "does not exist in the graph".
_VIOLATION_TYPES = [
    'R_STATUS', 'R_EMPTY', 'R_PENDING', 'R_REFINE', 'R_DUP',
    'R_ANSWER_ISOLATION', 'DEP', 'JSON_PARSE', 'R_PTP_JUSTIFICATION', 'OTHER',
]


def _classify_violation(error_msg: str) -> str:
    """Map a single error message to a rule family.

    Error sources:
    - _parse_plan_output: "No JSON block...", "JSON parse failed...",
      "JSON must contain 'node_statuses'..."
    - _validate_plan_structure: per-rule messages with stable keyword anchors.

    Uses strict prefix match (re.match) for R_REFINE because other messages
    mention "refine node" in non-initial positions (R_PENDING's "any refine
    node", R_DUP's "search or refine node").
    """
    if re.match(r'refine node\b', error_msg, re.IGNORECASE):
        return 'R_REFINE'

    patterns = [
        ('JSON_PARSE', r'No JSON block|JSON parse failed|JSON must contain'),
        ('R_STATUS', r'node_statuses must cover'),
        ('R_EMPTY', r'plan must contain at least one new node'),
        ('R_PENDING', r'not addressed by any refine'),
        ('R_ANSWER_ISOLATION', r'must be planned alone in its batch'),
        ('R_DUP', r'same description AND prompt'),
        ('R_PTP_JUSTIFICATION', r'is a new \(non-refine\) addition in patch mode'),
        ('DEP', r'depends on itself|depends on .* SAME batch|'
                r'depends on .* does not exist|depends on .* neither an existing'),
    ]
    for name, pat in patterns:
        if re.search(pat, error_msg, re.IGNORECASE):
            return name
    return 'OTHER'


def _split_and_classify(error_msg: str) -> List[str]:
    """Split a (possibly-compound) error message by `[Error N]` markers
    and classify each sub-error. Returns a list of violation types (>= 1)."""
    if '[Error ' in error_msg:
        parts = re.split(r'\[Error \d+\]\s*', error_msg)
        parts = [p.strip() for p in parts if p.strip()]
    else:
        parts = [error_msg]
    return [_classify_violation(p) for p in parts]


def _validate_plan_structure(
    plan_output,
    existing_node_ids: set,
    nodes_to_evaluate: Optional[set] = None,
    prior_search_signatures: Optional[Dict[str, set]] = None,
    existing_node_statuses: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Structural validation of a parsed plan batch.

    Rule order:

      Short-circuit (return immediately if violated):
        1. R-STATUS  AE ids fully covered
        2. R-EMPTY   nodes list non-empty

      Batch-level collect (all violations collected, reported together):
        3. R-PENDING Uncertain/Not Found with alive refine budget must be
              paired with refine. Exhausted lines are NOT enforced here
              (prompt encourages fallback, but model is free to defer).
        4. R-ANSWER-ISOLATION answer alone in batch

      Per-node collect (all violations collected, reported together):
        5a self-dep
        5b same-batch dep
        5c unknown dep
        5d R-REFINE: id matches '_refine1' or '_refine2'; parent must exist,
              parent status must be Uncertain or Not Found, refine number
              must be <= 2 (budget = 3 attempts: original + refine1 + refine2).
        5e R-DUP: any non-answer node (search OR refine) whose (description,
              prompt) signature byte-matches a prior node is rejected.

    Returns None if valid; otherwise returns an error string (multiple
    violations numbered [Error 1], [Error 2], ...) for the FORMAT WARNING
    retry loop.
    """
    new_nodes = plan_output.nodes
    errors = []  # Collect all violations, report them together.

    # 1. R-STATUS: every AE node needs an explicit verdict.
    if nodes_to_evaluate:
        provided = set((plan_output.node_statuses or {}).keys())
        missing = nodes_to_evaluate - provided
        if missing:
            return (
                f"node_statuses must cover every id listed under "
                f"'Nodes to evaluate this turn'. Missing: {sorted(missing)}. "
                f"Each Awaiting Evaluation node requires an explicit 'Success' / "
                f"'Uncertain' / 'Not Found' verdict every turn — silence is not "
                f"allowed. Re-output the JSON with all required ids included."
            )

    # 2. R-EMPTY: every iteration must produce at least one new node.
    if not new_nodes:
        return (
            "plan must contain at least one new node. Every iteration evaluates the "
            "previous turn's nodes and plans new ones; an empty 'nodes' array is not "
            "a valid iteration result. If all evaluated nodes are Success and the "
            "Termination conditions are met, plan a single answer node; otherwise plan "
            "the next batch of search/refine nodes."
        )


    new_ids = {n.id for n in new_nodes}

    # 3. R-PENDING: every Uncertain or Not Found set this turn must be
    # addressed in the same plan. Two legal ways:
    #   (a) line still alive -> pair with refine node
    #   (b) line exhausted -> prompt encourages fallback (not enforced)
    pending = {
        nid for nid, st in (plan_output.node_statuses or {}).items()
        if st in ("Uncertain", "Not Found")
    }
    if pending:
        # Collect the immediate parent id of each refine node in this batch.
        refine_parents = set()
        for n in new_nodes:
            if "_refine" in n.id:
                m = _REFINE_ID_RE.match(n.id)
                if m:
                    root_id = m.group(1)
                    refine_num = int(m.group(2))
                    # Immediate parent: root for refine1, previous refine for refine2
                    if refine_num == 1:
                        refine_parents.add(root_id)
                    else:
                        refine_parents.add(f"{root_id}_refine{refine_num - 1}")
        alive_uncovered = set()
        exhausted_uncovered = set()
        for nid in pending:
            if "_refine" in nid:
                m = _REFINE_ID_RE.match(nid)
                if m and int(m.group(2)) >= 2:
                    # _refine2: 3rd attempt, line exhausted
                    exhausted_uncovered.add(nid)
                elif nid not in refine_parents:
                    # _refine1: 2nd attempt, still has budget for refine2
                    alive_uncovered.add(nid)
                continue
            # Original search node
            if f"{nid}_refine1" in existing_node_ids:
                if f"{nid}_refine2" in existing_node_ids:
                    # All 3 attempts used, exhausted
                    exhausted_uncovered.add(nid)
                # else: edge case (original re-pending with refine1 existing)
                continue
            if nid not in refine_parents:
                alive_uncovered.add(nid)

        if alive_uncovered:
            errors.append(
                f"the following nodes were marked Uncertain or Not Found this turn "
                f"but are NOT addressed by any refine node in this same plan: "
                f"{sorted(alive_uncovered)}. Every Uncertain or Not Found node whose "
                f"line of inquiry still has refine budget left MUST be paired with a "
                f"refine node ('<parent>_refineN') in the same turn. "
                f"There is no 'silently abandon' option."
            )

        # Exhausted lines: do NOT enforce fallback in validator.
        # The prompt encourages fallback but the model is free to
        # address exhausted lines in the next turn after seeing
        # refine results, or to move on to checking conditions if candidates
        # have already been found.

    # 4. Answer-batch-isolation: answer node must be alone in batch.
    answer_nodes = [n for n in new_nodes if n.id.endswith("_answer")]
    if answer_nodes and len(new_nodes) > 1:
        other_ids = sorted(n.id for n in new_nodes if n.id != answer_nodes[0].id)
        errors.append(
            f"answer node '{answer_nodes[0].id}' must be planned alone in its batch, "
            f"but this batch also contains: {other_ids}. "
            f"Re-plan this turn: either remove the answer node and finish verification first, "
            f"or remove the other nodes and submit ONLY the answer node."
        )

    # Per-node loop: dependency rules + R-REFINE + R-DUP
    for n in new_nodes:
        # 5a-5c: dependency rules
        for dep in n.dependencies or []:
            if dep == n.id:
                errors.append(
                    f"node '{n.id}' depends on itself. Remove the self-dependency."
                )
            elif dep in new_ids:
                errors.append(
                    f"node '{n.id}' depends on '{dep}', which is being planned in the SAME batch "
                    f"and has not been evaluated yet. All dependencies must already exist in the graph "
                    f"with an evaluated status (Success / Uncertain / Not Found). "
                    f"Re-plan: either remove this dependency, or split the batch so that '{dep}' "
                    f"is planned and evaluated in an earlier turn before '{n.id}'."
                )
            elif dep not in existing_node_ids:
                errors.append(
                    f"node '{n.id}' depends on '{dep}', which does not exist in the graph. "
                    f"Use only ids of nodes already in the graph."
                )

        # R-REFINE: 3-attempt limit (original + refine1 + refine2)
        if "_refine" in n.id:
            m = _REFINE_ID_RE.match(n.id)
            if not m:
                errors.append(
                    f"refine node '{n.id}' has malformed id; expected "
                    f"'<parent>_refine1' or '<parent>_refine2'."
                )
            else:
                root_id = m.group(1)
                refine_num = int(m.group(2))
                if refine_num > 2:
                    errors.append(
                        f"refine node '{n.id}' exceeds the 3-attempt limit "
                        f"(original + refine1 + refine2). No further refines allowed."
                    )
                else:
                    # Compute immediate parent
                    if refine_num == 1:
                        parent_id = root_id
                    else:
                        parent_id = f"{root_id}_refine{refine_num - 1}"
                    if parent_id not in existing_node_ids:
                        errors.append(
                            f"refine node '{n.id}' references parent '{parent_id}' which does "
                            f"not exist in the graph. A refine node must be tied to a real "
                            f"parent that has been executed and evaluated."
                        )
                    else:
                        parent_status = (plan_output.node_statuses or {}).get(parent_id)
                        if not parent_status:
                            parent_status = (existing_node_statuses or {}).get(parent_id)
                        if parent_status not in ("Uncertain", "Not Found"):
                            errors.append(
                                f"refine node '{n.id}' parent '{parent_id}' has status "
                                f"'{parent_status}'. Only nodes with status 'Uncertain' or "
                                f"'Not Found' can be refined."
                            )
                        if n.id in existing_node_ids:
                            errors.append(
                                f"refine node '{n.id}' already exists in the graph. "
                                f"Each refine id can only be used once."
                            )

        # R-DUP
        if prior_search_signatures and not n.id.endswith("_answer"):
            desc_key = _normalize_search_text(n.description)
            prompt_key = _normalize_search_text(n.prompt)
            prior_prompts = prior_search_signatures.get(desc_key)
            if prior_prompts and prompt_key in prior_prompts:
                errors.append(
                    f"node '{n.id}' has the same description AND prompt as a previously-issued "
                    f"search or refine node. Repeating an identical query verbatim cannot yield "
                    f"new evidence. If you need to explore the same line of inquiry again, the "
                    f"prompt MUST differ from the prior attempt by explicitly listing (a) the "
                    f"specific queries the parent already attempted (from its key_steps) and "
                    f"(b) a new angle / source / keyword strategy you want this time."
                )

    if not errors:
        return None
    if len(errors) == 1:
        return errors[0]
    return "\n".join(f"[Error {i+1}] {e}" for i, e in enumerate(errors))


class Orchestrator:
    """Evaluate-then-Grow incremental planner with stateful Agent for RL trajectory."""

    # Context budget warning threshold. When context usage crosses this ratio,
    # a single one-shot warning is injected telling the model to start walking
    # the termination path (which already self-checks answer form before
    # emitting an answer node). The warning is advisory: the validator does
    # NOT force answer-only output, because the termination path itself may
    # legitimately plan an exact-form verification search/refine before the
    # final answer. The ratio leaves enough headroom for ~2 more orchestrator
    # turns at ~5-10% per turn (form-check + answer).
    BUDGET_WARN_RATIO = 0.85

    def __init__(self, llm_client, tokenizer, config, stats=None):
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.config = config
        self._node_counter = 0
        self.agent: Optional[Agent] = None
        self._system_prompt: Optional[str] = None
        self._last_status_changes: Dict[str, str] = {}
        self.stats = stats
        # One-shot budget warning latch. Becomes True after the warning is
        # injected so we do not re-inject it on subsequent turns.
        self._budget_warned: bool = False
        # C3 metrics: structural violation counters (fixed-schema dict)
        self._format_warning_count: int = 0
        self._format_warning_types: Dict[str, int] = {vt: 0 for vt in _VIOLATION_TYPES}

    async def plan(self, overall_goal: str, graph: TaskDAG,
                   ability: str = '', new_node_ids: List[str] = None) -> tuple:
        """
        Incremental planning with delta user prompts and JSON output.

        Args:
            overall_goal: The task to solve.
            graph: Current TaskDAG state.
            ability: Environment type (e.g. 'LocalSearch', 'WebSearch').
            new_node_ids: IDs of nodes newly executed since last plan() call.

        Returns:
            (PlanResult, llm_response_text)
        """
        # One plan() invocation = one Orchestrator decision round, regardless of
        # how many parse-retries the inner Agent.step() loop runs. The internal
        # Agent has count_step_key=None so Agent.step does not double-count.
        if self.stats is not None:
            self.stats['orchestrator_steps'] = self.stats.get('orchestrator_steps', 0) + 1
            self.stats['plan_calls'] = self.stats.get('plan_calls', 0) + 1

        from .prompts import (ORCHESTRATOR_SYSTEM_PROMPT, ORCHESTRATOR_INITIAL_USER_PROMPT,
                              ORCHESTRATOR_DELTA_USER_PROMPT, get_search_strategy)

        strategy = get_search_strategy(ability)

        if self.agent is None:
            # First call: system prompt with overall goal, initial full graph state
            system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
                overall_goal=overall_goal, **strategy)
            graph_state = graph.to_prompt_format() if graph else "[Empty graph - initial planning]"
            user_prompt = ORCHESTRATOR_INITIAL_USER_PROMPT.format(graph_state=graph_state)
        else:
            # Subsequent calls: delta + compact ledger + explicit AE node list
            system_prompt = None  # already set
            delta_results = graph.to_delta_update(
                new_node_ids or [], self._last_status_changes)
            compact_ledger = graph.to_compact_ledger()
            ae_nodes = [n.id for n in graph.nodes.values() if not n.status]
            ae_node_ids_str = ', '.join(ae_nodes) if ae_nodes else '(none — all nodes already evaluated)'
            user_prompt = ORCHESTRATOR_DELTA_USER_PROMPT.format(
                delta_results=delta_results, compact_ledger=compact_ledger,
                ae_node_ids=ae_node_ids_str)

        # Build validator inputs (only meaningful for subsequent calls).
        nodes_to_evaluate_set: set = set()
        prior_search_signatures: Dict[str, set] = {}
        existing_node_statuses: Dict[str, str] = {}
        if self.agent is not None and graph is not None:
            nodes_to_evaluate_set = {n.id for n in graph.nodes.values() if not n.status}
            existing_node_statuses = {nid: node.status for nid, node in graph.nodes.items()}
            # R-DUP signature pool: every prior non-answer node (search OR
            # refine) contributes its (description, prompt) signature so a
            # refine that copies its parent verbatim is rejected.
            for node in graph.nodes.values():
                if node.id.endswith("_answer"):
                    continue
                desc_key = _normalize_search_text(node.description)
                prompt_key = _normalize_search_text(getattr(node, "prompt", ""))
                prior_search_signatures.setdefault(desc_key, set()).add(prompt_key)

        if self.agent is None:
            init_messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            # count_step_key=None: parse-retries inside plan() should not inflate
            # orchestrator_steps. The +1 is issued exactly once per plan() call,
            # at the entry of Orchestrator.plan() below.
            self.agent = Agent(self.llm_client, init_messages, self.tokenizer,
                               self.config, prompt_turn=len(init_messages),
                               stats=self.stats, count_step_key=None)
            self._system_prompt = system_prompt
        else:
            self.agent.append({'role': 'user', 'content': user_prompt})

        # Token budget check helper (matches Agent.step() budget: prompt_ids_len + response_length)
        min_output_tokens = getattr(self.config.plugin, 'min_required_output_tokens', 256)

        def _check_budget():
            current_len = len(self.agent.context_ids())
            max_len = self.agent.prompt_ids_len + self.config.response_length
            remaining = max_len - current_len
            return remaining, current_len, max_len

        remaining, current_len, max_len = _check_budget()
        if remaining < min_output_tokens:
            print(f"[Orchestrator] Context exhausted: {current_len}/{max_len}, remaining={remaining}")
            self.rollback_last_turn()
            return PlanResult(status="context_exhausted",
                              diagnostics=f"context {current_len}/{max_len}, remaining={remaining}"), ""

        # Context budget warning: a single advisory message tells the model
        # to start walking the termination path. The termination path itself
        # already self-checks answer form (Answer Rule form check) and
        # may legitimately plan an exact-form verification search/refine
        # before the final answer node, so the validator does NOT force
        # answer-only output here.
        if self.agent is not None and not self._budget_warned:
            ratio = current_len / max_len if max_len > 0 else 0.0
            if ratio >= self.BUDGET_WARN_RATIO:
                self._budget_warned = True
                budget_warning = (
                    "[CONTEXT WARNING] Orchestrator context is at "
                    f"{ratio:.0%} of capacity (threshold {self.BUDGET_WARN_RATIO:.0%}). "
                    "Roughly 2 planning turns remain before the budget is exhausted. "
                    "Begin converging now: select the best-supported candidate from "
                    "the current ledger and follow the Answer Rule. If the Overall "
                    "Goal requires a specific form and no upstream node has retrieved "
                    "it yet, plan a form-verification search this turn, then the "
                    "answer node next turn. Otherwise plan the answer node directly."
                )
                self.agent.append({'role': 'user', 'content': budget_warning})
                print(f"[Orchestrator] Budget warning fired at "
                      f"{current_len}/{max_len} = {ratio:.1%}")

        # FORMAT WARNING loop: step() records assistant turn with real logprobs
        llm_response_text = ""
        last_error = ""
        for retry in range(4):
            # Re-check budget before each retry (warning turns consume tokens)
            if retry > 0:
                remaining, current_len, max_len = _check_budget()
                if remaining < min_output_tokens:
                    print(f"[Orchestrator] Context exhausted during retry: remaining={remaining}")
                    return PlanResult(status="context_exhausted",
                                     diagnostics=f"exhausted during retry {retry}"), llm_response_text

            try:
                response = await self.agent.step()
                if response is None:
                    self.rollback_last_turn()
                    return PlanResult(status="llm_no_response",
                                     diagnostics="agent.step() returned None"), ""
                llm_response_text = response
            except Exception as e:
                self.rollback_last_turn()
                return PlanResult(status="llm_exception",
                                 diagnostics=str(e)), ""

            plan_output, parse_error = self._parse_plan_output(llm_response_text)

            # Structural validation: if JSON parsed OK but the batch violates
            # batch-shape rules (e.g. answer-not-alone, same-batch deps),
            # treat it as a format error and retry via the FORMAT WARNING loop.
            if not parse_error:
                structure_error = _validate_plan_structure(
                    plan_output,
                    set(graph.nodes.keys()),
                    nodes_to_evaluate=nodes_to_evaluate_set,
                    prior_search_signatures=prior_search_signatures,
                    existing_node_statuses=existing_node_statuses,
                )
                if structure_error:
                    parse_error = structure_error

            # Successful parse with content
            if not parse_error and (plan_output.nodes or plan_output.node_statuses):
                self._last_status_changes = plan_output.node_statuses
                return PlanResult(status="ok", plan_output=plan_output,
                                  raw_text=llm_response_text), llm_response_text

            # Valid JSON parsed but empty content (no_progress, not a format error)
            if not parse_error:
                return PlanResult(status="no_progress", plan_output=plan_output,
                                  raw_text=llm_response_text,
                                  diagnostics="Valid plan but no nodes and no status updates"), llm_response_text

            # Parse error: classify violations, apply C3 penalty, retry
            last_error = parse_error
            # Classify each sub-error (handles compound [Error N] messages)
            violation_types = _split_and_classify(parse_error)
            self._format_warning_count += len(violation_types)
            for vt in violation_types:
                self._format_warning_types[vt] += 1
            enable_structural_penalty = getattr(self.config.plugin, 'dagent_structural_penalty', False)
            if enable_structural_penalty:
                current_turn_idx = len(self.agent.chat) - 1
                self.agent.set_process_reward(current_turn_idx, -1)
            if retry < 3:
                warning = f"[FORMAT WARNING] {parse_error} " \
                          "Output your thinking, then a single JSON object in a ```json block."
                self.agent.append({'role': 'user', 'content': warning})
                print(f"[Orchestrator] Format warning ({retry + 1}/3): {parse_error}")

        print(f"[Orchestrator] Format warning max retries (3) reached: {last_error}")
        return PlanResult(status="parse_error", raw_text=llm_response_text,
                          diagnostics=last_error), llm_response_text

    def _parse_plan_output(self, response_text: str) -> Tuple[PlanOutput, str]:
        """Parse plan JSON from LLM response. Returns (PlanOutput, error_string)."""
        empty = PlanOutput(node_statuses={}, nodes=[])

        # Extract JSON block
        json_text = _extract_json_block(response_text)
        if not json_text:
            return empty, "No JSON block found. Output a ```json block with node_statuses and nodes."

        # Parse JSON with fallback chain
        parsed = _parse_plan_json(json_text)
        if parsed is None:
            return empty, f"JSON parse failed. Check double quotes and escaping."

        # Schema validation
        if not _validate_plan_schema(parsed):
            return empty, "JSON must contain 'node_statuses' (object) and 'nodes' (array)."

        # Extract node_statuses
        valid_statuses = {'Success', 'Uncertain', 'Not Found', 'Failed'}
        node_statuses = {}
        raw_statuses = parsed.get('node_statuses', {})
        if isinstance(raw_statuses, dict):
            node_statuses = {k: v for k, v in raw_statuses.items() if v in valid_statuses}

        # Extract nodes (permissive: auto-fill missing id/dependencies)
        nodes = []
        raw_nodes = parsed.get('nodes', [])
        if isinstance(raw_nodes, list):
            for n in raw_nodes:
                if isinstance(n, dict) and 'description' in n and 'prompt' in n:
                    node = TaskNodeSpec.from_dict(n)
                    if not node.id:
                        self._node_counter += 1
                        node.id = f"t{self._node_counter}"
                    nodes.append(node)

        return PlanOutput(node_statuses=node_statuses, nodes=nodes), ""

    def rollback_last_turn(self):
        """Rollback the Agent's last turn (for step failure or parse failure)."""
        if self.agent and len(self.agent.chat) > self.agent.prompt_turn:
            self.agent.chat.pop()
            self.agent.chat_completions.pop()
            self.agent.additional_info.pop()
            self.agent.chat_ids.pop()
            self.agent.log_probs.pop()
            self.agent.token_mask.pop()


# ============================================================================
# Execution - parallel executor invocations
# ============================================================================

def extract_all_fn_calls(text):
    """Extract all function calls from a model response."""
    if text is None:
        return []
    calls = []
    pattern = r'<function=([^>]+)>(.*?)</function>'
    matches = re.finditer(pattern, text, re.DOTALL)
    for match in matches:
        fn_name = match.group(1)
        fn_content = match.group(2)
        params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', fn_content, re.DOTALL))
        calls.append({'function': fn_name, 'arguments': params})
    return calls


async def execute_nodes(
    nodes: list,
    graph: TaskDAG,
    env,
    llm_client,
    tokenizer,
    recall_tool: RecallTool,
    max_turn: int,
    task_timeout: int,
    stats: dict,
) -> tuple:
    """
    Execute a list of nodes in parallel.

    Returns:
        (result_map, batch_node_metrics)
    """
    if not nodes:
        return {}, []

    print(f'[DAGent] Executing {len(nodes)} nodes in parallel: {[n.id for n in nodes]}')

    tasks = []
    for node in nodes:
        upstream_context = graph.get_upstream_context(node.id)

        task = _execute_single_node(
            node=node,
            upstream_context=upstream_context,
            env=env,
            llm_client=llm_client,
            tokenizer=tokenizer,
            recall_tool=recall_tool,
            max_turn=max_turn,
            task_timeout=task_timeout,
            stats=stats,
        )
        tasks.append((node.id, task))

    results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

    result_map = {}
    batch_node_metrics = []
    for (node_id, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            graph.update_node(
                node_id,
                status="Failed",
                answer=f"Error: {str(result)}",
                explanation="",
                key_steps='',
                confidence="0%",
                uncertainties=f"Execution error: {str(result)}",
                transcript=""
            )
            result_map[node_id] = {
                'status': 'Failed',
                'error': str(result),
                'failure_reason': f'exception: {str(result)}',
                'finish_output': {},
                'turns': 0,
                'executor_llm_calls': 0,
                'messages': [],
                'completions': [],
                'transcript': '',
            }
            node = graph.get_node(node_id)
            if node:
                batch_node_metrics.append({
                    'node_id': node_id,
                    'depth': node.depth,
                    'num_dependencies': len(node.dependencies),
                    'prompt_tokens': 0,
                    'final_prompt_tokens': 0,
                    'upstream_tokens': 0,
                    'dep_ratio': 0,
                    'overflow_flag': False,
                    'truncated_tokens': 0,
                    'status': 'exception'
                })
        else:
            finish_output = result.get('finish_output', {})
            failure_reason = result.get('failure_reason')
            status = 'Failed' if failure_reason else ''

            graph.update_node(
                node_id,
                status=status,
                answer=finish_output.get('answer', ''),
                explanation=finish_output.get('explanation', ''),
                key_steps=finish_output.get('key_steps', ''),
                confidence=finish_output.get('confidence', '0%'),
                uncertainties=finish_output.get('uncertainties', ''),
                transcript=result.get('transcript', '')
            )
            result_map[node_id] = {
                'status': status,
                'finish_output': finish_output,
                'turns': result.get('turns', 0),
                'executor_llm_calls': result.get('executor_llm_calls', 0),
                'failure_reason': failure_reason,
                'messages': result.get('messages', []),
                'completions': result.get('completions', []),
                'transcript': result.get('transcript', ''),
            }
            if 'node_metrics' in result:
                batch_node_metrics.append(result['node_metrics'])

    return result_map, batch_node_metrics


async def _execute_single_node(
    node: TaskNode,
    upstream_context: str,
    env,
    llm_client,
    tokenizer,
    recall_tool: RecallTool,
    max_turn: int,
    task_timeout: int,
    stats: dict,
) -> dict:
    """Run the ReAct loop for a single TaskNode."""
    item = {
        'upstream_context': upstream_context,
        'task_description': node.description,
        'task_prompt': node.prompt,
    }
    chat = create_chat(node.description, workflow='dagent', item=item)

    messages = chat.copy()
    completions = []  # Parallel to messages[2:]; assistant turn → completion, user turn → None
    transcript_lines = []
    finish_output = {}
    failure_reason = None
    start_time = time.time()
    format_warning_count = {'finish': 0, 'recall': 0}

    initial_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **_chat_template_kwargs())
    prompt_tokens = len(initial_ids)
    upstream_tokens = len(tokenizer.encode(upstream_context)) if upstream_context else 0
    overflow_flag = False
    truncated_tokens = 0
    local_executor_llm_calls = 0  # plan §4.5: completed LLM forward count (excluding empty retries)

    for turn in range(max_turn):
        elapsed = time.time() - start_time
        if elapsed > task_timeout:
            failure_reason = 'timeout'
            break

        # Invoke LLM (with response normalization and exponential backoff).
        response_text = None
        max_empty_retries = 10
        last_error = None
        for empty_retry in range(max_empty_retries):
            try:
                input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **_chat_template_kwargs())
                config = llm_client.config
                max_len = len(input_ids) + config.response_length
                response = await llm_client.create_completion(
                    input_ids=input_ids, messages=messages, max_len=max_len,
                    stats=stats
                )

                if response is None:
                    failure_reason = 'llm_error: no response from server'
                    break

                local_executor_llm_calls += 1  # plan §4.5: count only completed forwards
                response_text = response['choices'][0]['message']['content']
                if response_text and response_text.strip():
                    break
                else:
                    wait_time = min(2 ** empty_retry, 30)
                    print(f"[ReAct] Empty response, retry {empty_retry + 1}/{max_empty_retries}, waiting {wait_time}s...")
                    if empty_retry < max_empty_retries - 1:
                        await asyncio.sleep(wait_time)
            except ContextOverflowError as e:
                overflow_flag = True
                failure_reason = f'context_overflow: {e}'
                print(f"[ReAct] Context overflow, stopping node: {e}")
                break
            except Exception as e:
                last_error = str(e)
                wait_time = min(2 ** empty_retry, 30)
                print(f"[ReAct] LLM error: {last_error}, retry {empty_retry + 1}/{max_empty_retries}, waiting {wait_time}s...")
                if empty_retry < max_empty_retries - 1:
                    await asyncio.sleep(wait_time)
                continue

        if failure_reason or not response_text:
            if not failure_reason:
                if last_error:
                    failure_reason = f'llm_error: {last_error} (after {max_empty_retries} retries)'
                else:
                    failure_reason = f'llm_error: empty response after {max_empty_retries} retries'
            break

        messages.append({'role': 'assistant', 'content': response_text})
        completions.append(response)  # Save full completion with raw_output_ids/logprobs
        transcript_lines.append(f"[Turn {turn + 1}]\n{response_text}")

        fn_calls = extract_all_fn_calls(response_text)
        if not fn_calls:
            continue

        # Check for finish.
        finish_format_error = False
        for fn_call in fn_calls:
            if fn_call['function'] == 'finish':
                args = fn_call['arguments']
                required_finish_fields = ['answer', 'explanation', 'key_steps', 'confidence', 'uncertainties']
                if not all(f in args for f in required_finish_fields):
                    format_warning_count['finish'] += 1
                    missing = [f for f in required_finish_fields if f not in args]
                    if format_warning_count['finish'] <= 3:
                        messages.append({'role': 'user', 'content':
                            f"[FORMAT WARNING] Missing fields in finish: {missing}. "
                            "Please include all required fields in your finish call."
                        })
                        completions.append(None)  # user turn
                        transcript_lines.append(f"[Format Warning] finish format error ({format_warning_count['finish']}/3)")
                        finish_format_error = True
                        break
                    transcript_lines.append("[Format Warning] Max retries (3) reached, using partial output as fallback")

                finish_output = {
                    'answer': args.get('answer', ''),
                    'explanation': args.get('explanation', ''),
                    'key_steps': args.get('key_steps', ''),
                    'confidence': args.get('confidence', '0%'),
                    'uncertainties': args.get('uncertainties', '')
                }

                final_prompt_tokens = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, **_chat_template_kwargs()))
                node_metrics = {
                    'node_id': node.id,
                    'depth': node.depth,
                    'num_dependencies': len(node.dependencies),
                    'prompt_tokens': prompt_tokens,
                    'final_prompt_tokens': final_prompt_tokens,
                    'upstream_tokens': upstream_tokens,
                    'dep_ratio': upstream_tokens / prompt_tokens if prompt_tokens > 0 else 0,
                    'overflow_flag': overflow_flag,
                    'truncated_tokens': truncated_tokens,
                    'status': 'completed'
                }

                assert len(completions) == len(messages) - 2, \
                    f"completions/messages misaligned: {len(completions)} vs {len(messages)-2}"
                return {
                    'finish_output': finish_output,
                    'transcript': '\n\n'.join(transcript_lines),
                    'turns': turn + 1,
                    'executor_llm_calls': local_executor_llm_calls,
                    'messages': messages,
                    'completions': completions,
                    'node_metrics': node_metrics
                }

        if finish_format_error:
            continue

        # Handle other tool calls.
        observation_parts = []

        # Handle recall.
        for fn_call in fn_calls:
            if fn_call['function'] == 'recall':
                args = fn_call['arguments']
                if 'node_id' not in args or 'goal' not in args:
                    format_warning_count['recall'] += 1
                    missing_recall = []
                    if 'node_id' not in args: missing_recall.append('node_id')
                    if 'goal' not in args: missing_recall.append('goal')
                    if format_warning_count['recall'] <= 3:
                        observation_parts.append(
                            f"[FORMAT WARNING] Missing fields in recall: {missing_recall}. "
                            "Please include node_id and goal in your recall call."
                        )
                    else:
                        observation_parts.append("[Recall skipped: format error persists after 3 warnings]")
                    continue
                recall_result = await recall_tool.call(args.get('node_id', ''), args.get('goal', ''))
                observation_parts.append(recall_result)
                stats['recall_calls'] += 1

        # Handle search / open_page (delegated to env).
        has_env_calls = any(fn['function'] in ('search', 'open_page') for fn in fn_calls)
        if has_env_calls:
            for fn in fn_calls:
                if fn['function'] == 'search':
                    stats['search_calls'] += 1
                elif fn['function'] == 'open_page':
                    stats['open_page_calls'] += 1
            observation = await run_action(env, response_text)
            if observation is None:
                if hasattr(env, 'predicted_answer') and env.predicted_answer:
                    finish_output = {
                        'answer': env.predicted_answer[0],
                        'explanation': env.predicted_answer[1] or '',
                        'key_steps': '',
                        'confidence': env.predicted_answer[2] or '0%',
                        'uncertainties': ''
                    }
                    final_prompt_tokens = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, **_chat_template_kwargs()))
                    node_metrics = {
                        'node_id': node.id,
                        'depth': node.depth,
                        'num_dependencies': len(node.dependencies),
                        'prompt_tokens': prompt_tokens,
                        'final_prompt_tokens': final_prompt_tokens,
                        'upstream_tokens': upstream_tokens,
                        'dep_ratio': upstream_tokens / prompt_tokens if prompt_tokens > 0 else 0,
                        'overflow_flag': overflow_flag,
                        'truncated_tokens': truncated_tokens,
                        'status': 'completed'
                    }
                    assert len(completions) == len(messages) - 2, \
                        f"completions/messages misaligned: {len(completions)} vs {len(messages)-2}"
                    return {
                        'finish_output': finish_output,
                        'transcript': '\n\n'.join(transcript_lines),
                        'turns': turn + 1,
                        'executor_llm_calls': local_executor_llm_calls,
                        'messages': messages,
                        'completions': completions,
                        'node_metrics': node_metrics
                    }
            else:
                observation_parts.append(observation)

        if observation_parts:
            full_observation = '\n\n'.join(observation_parts)

            # Pre-check whether appending the next result would exceed the context limit (95% threshold).
            temp_messages = messages + [{'role': 'user', 'content': full_observation}]
            temp_ids = tokenizer.apply_chat_template(temp_messages, add_generation_prompt=True, **_chat_template_kwargs())
            max_context = llm_client.config.prompt_length + llm_client.config.response_length

            if len(temp_ids) > max_context * 0.95:
                overflow_flag = True
                truncated_tokens += len(tokenizer.encode(full_observation))
                context_warning = """[CONTEXT WARNING] The search results would exceed the context limit (>95%).
The results have been discarded. Please use the `finish` tool NOW to provide your best-effort answer based on current information.
Include any unresolved questions in the `uncertainties` field."""
                messages.append({'role': 'user', 'content': context_warning})
                completions.append(None)  # user turn
                transcript_lines.append(f"[Context Warning] Results discarded due to context limit")
            else:
                messages.append({'role': 'user', 'content': full_observation})
                completions.append(None)  # user turn
                transcript_lines.append(f"[Observation]\n{full_observation[:2000]}...")

    # Did not finish normally.
    if not finish_output:
        if failure_reason is None:
            failure_reason = 'max_turn'
        finish_output = {
            'answer': f'Task incomplete - {failure_reason}',
            'explanation': f'Task did not complete due to {failure_reason}',
            'key_steps': '',
            'confidence': '0%',
            'uncertainties': f'Task incomplete: {failure_reason}'
        }

    final_prompt_tokens = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, **_chat_template_kwargs()))
    node_metrics = {
        'node_id': node.id,
        'depth': node.depth,
        'num_dependencies': len(node.dependencies),
        'prompt_tokens': prompt_tokens,
        'final_prompt_tokens': final_prompt_tokens,
        'upstream_tokens': upstream_tokens,
        'dep_ratio': upstream_tokens / prompt_tokens if prompt_tokens > 0 else 0,
        'overflow_flag': overflow_flag,
        'truncated_tokens': truncated_tokens,
        'status': 'failed' if failure_reason else 'completed'
    }

    assert len(completions) == len(messages) - 2, \
        f"completions/messages misaligned: {len(completions)} vs {len(messages)-2}"
    return {
        'finish_output': finish_output,
        'transcript': '\n\n'.join(transcript_lines),
        'turns': turn + 1 if 'turn' in locals() else 0,
        'executor_llm_calls': local_executor_llm_calls,
        'failure_reason': failure_reason,
        'messages': messages,
        'completions': completions,
        'node_metrics': node_metrics
    }


# ============================================================================
# Entry Point - DAGent Evaluate-then-Grow Workflow
# ============================================================================

async def process_item(
    item: DataProto,
    context: TaskContext,
) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
    """
    DAGent workflow entry point.
    Runs Evaluate-then-Grow incremental planning with parallel sub-task execution.

    Returns:
        list[AgentLoopOutput]: Orchestrator trajectory (index 0) + executor node trajectories.
    """
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)

    # Fail-fast: DAGent requires sufficient response_length for multi-turn Orchestrator
    min_response_length = getattr(config.plugin, 'dagent_min_response_length', 8192)
    if config.response_length < min_response_length:
        print(f"[DAGent WARNING] response_length={config.response_length} is below recommended "
              f"minimum {min_response_length} for DAGent. Risk of context exhaustion.")

    ability = item.non_tensor_batch['ability'][0]

    uid = item.non_tensor_batch['uid'][0] if 'uid' in item.non_tensor_batch else uuid4().hex
    gen_uid = item.non_tensor_batch['gen_uid'][0] if 'gen_uid' in item.non_tensor_batch else None

    EnvClass = select_env(ability, config)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[DAGent Error] during environment init: {str(e)}")

    await env.get_data(item, context)

    problem_statement = env.instance_info['problem_statement']

    # Configuration parameters
    max_iterations = getattr(config.plugin, 'dagent_max_iterations', 30)
    task_timeout = getattr(config.plugin, 'dagent_task_timeout', 600)
    session_timeout = getattr(config.plugin, 'session_timeout', 5400)
    max_turn = getattr(config.plugin, 'val_max_turn', 200)
    max_traj = getattr(config.plugin, "max_traj", None)

    llm_client = context.llm_client

    # The stats dict must be initialized before instantiating the Orchestrator (plan §4.1).
    stats = {
        'iterations': 0,
        'total_turns': 0,
        'nodes_executed': 0,
        'parallel_batches': 0,
        'recall_calls': 0,
        'search_calls': 0,
        'open_page_calls': 0,
        'start_time': time.time(),
        # Efficiency instrumentation (plan §4.1)
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'external_tool_calls': 0,
        'orchestrator_steps': 0,
        'executor_batch_steps': 0,
        'plan_calls': 0,  # one increment per Orchestrator.plan() invocation,
                          # tracked separately from orchestrator_steps so that
                          # tool-call aggregation can include plan as one of the
                          # four DAGent agent-visible invocation types.
        # Per-iteration plan accounting.
        'initial_plan_nodes': 0,         # iter 0 plan output node count
        'initial_plan_roots': 0,         # iter 0 nodes with empty deps
        'initial_plan_depth': 0,         # iter 0 max depth
        'grow_iterations': 0,            # iter 1+ Orchestrator plan() count
        'nodes_added_per_iteration': [], # list of new-node counts per iter (>=1)
        'execution_instances': 0,        # executor invocations
        'unique_nodes_executed': 0,      # de-duplicated by node id
        'reexecuted_node_instances': 0,  # execution_instances - unique_nodes_executed
    }

    # Initialize components
    graph = TaskDAG()
    orchestrator = Orchestrator(llm_client, tokenizer, config, stats=stats)
    recall_tool = RecallTool(graph, llm_client, tokenizer)

    # Executor-node Agent objects (used for RL trajectories).
    executor_agents = {}

    all_messages = []
    all_node_metrics = []
    all_node_messages = {}
    session_start = time.time()

    print(f'[DAGent] Starting Evaluate-then-Grow workflow for: {problem_statement}')

    # ========== DAGent Evaluate-then-Grow main loop ==========
    # ========== DAGent main loop ==========
    last_batch_node_ids = []
    no_progress_count = 0
    llm_no_response_count = 0

    for iteration in range(max_iterations):
        if time.time() - session_start > session_timeout:
            print('[DAGent] Session Timeout')
            break

        stats['iterations'] = iteration + 1

        print(f"\n{'='*60}")
        print(f"[DAGent] Iteration {iteration + 1}")
        print(f"{'='*60}")

        # 1. Orchestrator incremental planning
        plan_result, orchestrator_response = await orchestrator.plan(
            overall_goal=problem_statement, graph=graph, ability=ability,
            new_node_ids=last_batch_node_ids
        )

        # Handle typed PlanResult
        if plan_result.status == "context_exhausted":
            print(f'[DAGent] Context exhausted, terminating: {plan_result.diagnostics}')
            break
        elif plan_result.status == "llm_no_response":
            llm_no_response_count += 1
            print(f'[DAGent] LLM no response ({llm_no_response_count}/3): {plan_result.diagnostics}')
            if llm_no_response_count >= 3:
                print('[DAGent] LLM no response limit reached, terminating')
                break
            continue
        elif plan_result.status == "llm_exception":
            print(f'[DAGent] LLM exception, terminating: {plan_result.diagnostics}')
            break
        elif plan_result.status == "parse_error":
            print(f'[DAGent] Parse error after retries: {plan_result.diagnostics}')
            # parse_error already retried 3 times inside plan(), don't retry externally
            break
        elif plan_result.status == "no_progress":
            # Valid plan but empty — handle in the no_progress section below
            pass

        # Reset llm_no_response counter on successful call
        llm_no_response_count = 0
        plan_output = plan_result.plan_output

        # Update node statuses from Orchestrator evaluation
        if last_batch_node_ids:
            for node_id, status in plan_output.node_statuses.items():
                if node_id in last_batch_node_ids and graph.get_node(node_id):
                    graph.update_node(node_id, status=status)
                    print(f"[Orchestrator] Updated node {node_id} status to: {status}")

        all_messages.append({
            'role': 'orchestrator',
            'iteration': iteration + 1,
            'response': orchestrator_response,
            'plan': {
                'node_statuses': plan_output.node_statuses,
                'nodes': [{'id': n.id, 'description': n.description,
                           'prompt': n.prompt, 'dependencies': n.dependencies}
                          for n in plan_output.nodes],
            }
        })

        print(f'[Orchestrator] Planned {len(plan_output.nodes)} nodes')

        # 2. Add new nodes to graph
        new_nodes = []
        for node_spec in plan_output.nodes:
            node = TaskNode(
                id=node_spec.id, description=node_spec.description,
                prompt=node_spec.prompt, dependencies=node_spec.dependencies
            )
            graph.add_node(node)
            new_nodes.append(node)
            deps_str = f" (deps: {node.dependencies})" if node.dependencies else ""
            print(f'[DAGent] Added node {node.id}{deps_str}:')
            print(f'    Description: {node.description}')

        # 3. Handle no new nodes
        if not new_nodes:
            answer_node_check = graph.get_answer_node()
            if answer_node_check and answer_node_check.answer:
                print('[DAGent] Answer node already completed, terminating')
                break
            # Evaluation-only round (statuses updated but no new nodes) is valid progress
            if plan_output.node_statuses:
                print(f'[DAGent] Evaluation-only round: updated {len(plan_output.node_statuses)} statuses')
                no_progress_count = 0
                last_batch_node_ids = []  # Clear so next delta doesn't resend old results
            else:
                no_progress_count += 1
                print(f'[DAGent] No new nodes and no status updates (no_progress {no_progress_count}/3)')
                if no_progress_count >= 3:
                    print('[DAGent] No progress limit reached, terminating')
                    break
            continue

        # Reset no_progress counter on successful planning
        no_progress_count = 0

        # 4. Execute all new nodes in parallel.
        stats['parallel_batches'] += 1
        result_map, batch_node_metrics = await execute_nodes(
            nodes=new_nodes, graph=graph, env=env,
            llm_client=llm_client, tokenizer=tokenizer,
            recall_tool=recall_tool, max_turn=max_turn,
            task_timeout=task_timeout, stats=stats,
        )
        all_node_metrics.extend(batch_node_metrics)

        # plan §4.5: batch wall-clock = max_i K_i
        batch_max = max(
            (r.get('executor_llm_calls', 0) for r in result_map.values()),
            default=0
        )
        stats['executor_batch_steps'] += batch_max

        for node_id, result in result_map.items():
            if 'messages' in result:
                all_node_messages[node_id] = result['messages']
                # Create the executor Agent for training (pass completion to recover real logprobs).
                if is_train and result.get('messages'):
                    exec_agent = Agent(llm_client, result['messages'][:2], tokenizer, config,
                                       prompt_turn=2, stats=stats, count_step_key=None)
                    node_completions = result.get('completions', [])
                    for idx, msg in enumerate(result['messages'][2:]):
                        comp = node_completions[idx] if idx < len(node_completions) else None
                        exec_agent.append(msg, completion=comp)
                    executor_agents[node_id] = exec_agent

        for node_id, result in result_map.items():
            stats['nodes_executed'] += 1
            stats['total_turns'] += result.get('turns', 0)
            finish_output = result.get('finish_output', {})
            failure_reason = result.get('failure_reason')
            status_display = f"Failed: {failure_reason}" if failure_reason else f"confidence={finish_output.get('confidence', '0%')}"
            print(f"[ReAct] Node {node_id} completed ({status_display}):")
            print(f"    Answer: {finish_output.get('answer', '')}")

        last_batch_node_ids = list(result_map.keys())

        all_messages.append({
            'role': 'execution',
            'iteration': iteration + 1,
            'nodes_executed': list(result_map.keys()),
            'results': {nid: {'status': r.get('status', '')} for nid, r in result_map.items()}
        })

        # 5. Termination check (the answer node has been executed).
        answer_node = graph.get_answer_node()
        if answer_node and answer_node.answer:
            print('[DAGent] Answer node executed, terminating')
            break

    # ========== Compile final results ==========
    stats['end_time'] = time.time()
    stats['duration'] = stats['end_time'] - stats['start_time']

    # plan §4.5: DAGent steps proxy = Orchestrator step + executor batch wall-clock max
    stats['wall_clock_steps_proxy'] = stats.get('orchestrator_steps', 0) + stats.get('executor_batch_steps', 0)
    # plan §4.6: external tool calls (recall excluded)
    stats['external_tool_calls'] = stats.get('search_calls', 0) + stats.get('open_page_calls', 0)

    stats['execution_instances'] = stats.get('nodes_executed', 0)
    stats['unique_nodes_executed'] = stats.get('nodes_executed', 0)
    stats['reexecuted_node_instances'] = 0
    # grow_iterations counts iter 1+ plan calls (plan calls after the initial growth seed).
    stats['grow_iterations'] = max(0, stats.get('plan_calls', 0) - 1)
    # refine_executions: number of refine nodes that landed in the graph,
    # semantically the right "re-attempt" count given DAGent's refine creates
    # a new node id rather than re-executing an existing id. Refine fires
    # when the Orchestrator marks a node Uncertain / Not Found and pairs it
    # with a refine in the same plan turn.
    try:
        stats['refine_executions'] = sum(1 for nid in graph.nodes if '_refine' in nid)
    except Exception:
        stats['refine_executions'] = 0

    final_answer = graph.get_final_answer()
    answer_node = graph.get_answer_node()

    try:
        if answer_node:
            env.predicted_answer = (answer_node.answer, answer_node.explanation, answer_node.confidence)
        elif final_answer:
            env.predicted_answer = (final_answer, "", "")
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, [], context), timeout=60 * 10)
        score = (score_msg, reward)
        print(f'[DAGent] Final Score: {score}')
    except Exception as e:
        print(f"[DAGent Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0}

    stats['score'] = score[1]
    stats['total_nodes'] = len(graph)

    try:
        max_depth = graph.compute_depths()
    except Exception as e:
        print(f"[DAGent] Depth computation failed, setting max_depth=0: {e}")
        max_depth = 0
    stats['max_depth'] = max_depth

    for metric in all_node_metrics:
        node = graph.get_node(metric['node_id'])
        if node:
            metric['depth'] = node.depth

    # ========== Compile AgentLoopOutput ==========
    # DAGRPO C2 config: multiplicative off-chain attenuation.
    # On-chain executor reward = R(τ); off-chain reward = α·R(τ) with α ∈ [0,1].
    # When R(τ)=0 the C2 signal collapses on that trajectory by design — failed
    # trajectories carry no on/off-chain credit signal because off-chain
    # executors there may carry unintegrated useful information rather than
    # wasted exploration, and a constant subtractive penalty would push the
    # policy away from such queries unfairly.
    alpha = getattr(config.plugin, 'dagent_credit_alpha', 1.0)

    # Answer chain for topology-conditioned reward shaping (C2)
    if answer_node is not None:
        answer_chain = graph.get_transitive_dependencies(answer_node.id)
    else:
        answer_chain = set()
        alpha = 1.0  # fallback: no on/off-chain split → no C2 differentiation

    # C3 metrics: structural violation counters from Orchestrator
    stats['format_warning_count'] = orchestrator._format_warning_count
    stats['format_warning_rate'] = (
        orchestrator._format_warning_count / max(stats.get('iterations', 0), 1)
    )
    for vt in _VIOLATION_TYPES:
        stats[f'format_warning_{vt}'] = orchestrator._format_warning_types[vt]

    # Construction-layer C2 stats (from graph.nodes; works for both train and eval)
    num_executor_nodes = len(graph.nodes)
    num_chain_nodes = sum(1 for nid in graph.nodes if nid in answer_chain)
    num_off_chain_nodes = num_executor_nodes - num_chain_nodes
    chain_ratio = num_chain_nodes / max(num_executor_nodes, 1)
    deps_total = sum(len(node.dependencies or []) for node in graph.nodes.values())
    avg_dependency_count = deps_total / max(num_executor_nodes, 1)
    answer_dep_count = len(answer_node.dependencies or []) if answer_node is not None else 0

    stats['num_executor_nodes'] = num_executor_nodes
    stats['num_chain_nodes'] = num_chain_nodes
    stats['num_off_chain_nodes'] = num_off_chain_nodes
    stats['chain_ratio'] = chain_ratio
    stats['avg_dependency_count'] = avg_dependency_count
    stats['answer_dependency_count'] = answer_dep_count

    # Training-layer C2 fields (default 0 for eval; overwritten after max_traj sampling)
    stats['num_executors_kept_for_training'] = 0
    stats['num_chain_executors_kept'] = 0
    stats['num_off_chain_executors_kept'] = 0
    stats['chain_truncated_count'] = 0

    outs = []

    # 1. Orchestrator trajectory (index 0, always kept).
    if orchestrator.agent is None:
        # Fallback: plan() was never called, create minimal agent for empty trajectory
        from .prompts import ORCHESTRATOR_SYSTEM_PROMPT, get_search_strategy
        strategy = get_search_strategy(ability)
        fallback_system = ORCHESTRATOR_SYSTEM_PROMPT.format(
            overall_goal=problem_statement, **strategy)
        orchestrator.agent = Agent(llm_client, [{'role': 'system', 'content': fallback_system}],
                                   tokenizer, config, prompt_turn=1,
                                   stats=stats, count_step_key=None)
    orch_out = await orchestrator.agent.get_data()
    outs.append(AgentLoopOutput(
        prompt_ids=orch_out['prompt_ids'],
        response_ids=orch_out['response_ids'],
        response_mask=orch_out['response_mask'],
        response_logprobs=orch_out['response_logprobs'],
        multi_modal_data={},
        reward_score=score[1],
        num_turns=orch_out['num_turns'],
        metrics=AgentLoopMetrics(),
        extra_fields={
            'messages': orch_out['messages'],
            'dagent_stats': copy.deepcopy(stats),
            'graph': graph.to_dict(),
            'node_metrics': all_node_metrics,
            'node_messages': all_node_messages,
            'orchestrator_log': all_messages,
            'is_finish': bool(answer_node and answer_node.answer),
            'agent_name': 'orchestrator',
            'uid': uid,
            'gen_uid': gen_uid,
            'role_uid': f'{uid}_orch',
            'process_reward_mask': orch_out.get('process_reward_mask', [0] * len(orch_out['response_mask'])),
            'mask_rollout': False,
        }
    ))

    # 2. Executor node trajectories (included during training).
    if is_train:
        for node_id, node_agent in executor_agents.items():
            try:
                node_out = await node_agent.get_data()
                on_chain = node_id in answer_chain
                exec_reward = score[1] if on_chain else alpha * score[1]
                # Executor must carry explicit zero mask (prevent concat crash in agent_loop.py:656)
                exec_prm = node_out.get('process_reward_mask')
                if exec_prm is None:
                    exec_prm = [0] * len(node_out['response_mask'])
                outs.append(AgentLoopOutput(
                    prompt_ids=node_out['prompt_ids'],
                    response_ids=node_out['response_ids'],
                    response_mask=node_out['response_mask'],
                    response_logprobs=node_out['response_logprobs'],
                    multi_modal_data={},
                    reward_score=exec_reward,
                    num_turns=node_out['num_turns'],
                    metrics=AgentLoopMetrics(),
                    extra_fields={
                        'messages': node_out['messages'],
                        'agent_name': node_id,
                        'uid': uid,
                        'gen_uid': gen_uid,
                        'role_uid': f'{uid}_exec',
                        'on_answer_chain': on_chain,
                        'process_reward_mask': exec_prm,
                        'mask_rollout': False,
                    }
                ))
            except Exception as e:
                print(f"[DAGent] Failed to get data for executor {node_id}: {e}")

    # Pre-sample chain count (before max_traj truncation) for chain_truncated_count
    _pre_sample_chain_count = sum(
        1 for o in outs[1:] if o.extra_fields.get('on_answer_chain')
    )

    # 3. Trajectory sampling (prioritize keeping answer-chain executors).
    if max_traj is not None and len(outs) > max_traj:
        chain_indices = [i for i in range(1, len(outs))
                         if outs[i].extra_fields.get('on_answer_chain')]
        # Deeper nodes first (closer to answer = more important)
        chain_indices.sort(
            key=lambda i: getattr(graph.get_node(outs[i].extra_fields.get('agent_name', '')), 'depth', 0),
            reverse=True)
        off_chain_indices = [i for i in range(1, len(outs)) if i not in set(chain_indices)]

        budget = max_traj - 1  # slot 0 = orchestrator
        keep = chain_indices[:budget]
        remaining = budget - len(keep)
        if remaining > 0 and off_chain_indices:
            keep += random.sample(off_chain_indices, k=min(remaining, len(off_chain_indices)))

        idx = [0] + sorted(keep)
        outs = [outs[i] for i in idx]

    # Training-layer C2 stats (after max_traj sampling)
    if is_train and len(outs) > 0:
        kept_exec_outs = outs[1:]  # exclude orchestrator
        num_executors_kept = len(kept_exec_outs)
        num_chain_kept = sum(
            1 for o in kept_exec_outs if o.extra_fields.get('on_answer_chain')
        )
        num_off_chain_kept = num_executors_kept - num_chain_kept

        stats['num_executors_kept_for_training'] = num_executors_kept
        stats['num_chain_executors_kept'] = num_chain_kept
        stats['num_off_chain_executors_kept'] = num_off_chain_kept
        stats['chain_truncated_count'] = max(0, _pre_sample_chain_count - num_chain_kept)

        # C2 reward-level evidence. Based on kept_exec_outs (the credit
        # assignment actually entering training), not the pre-sampling
        # outs[1:]. credit_alpha + has_answer_node flags distinguish
        # alpha<1 normal vs alpha=1 baseline vs answer_node-absent fallback.
        stats['credit_alpha'] = float(alpha)
        stats['has_answer_node'] = int(answer_node is not None)

        if kept_exec_outs:
            exec_rewards = [float(o.reward_score) for o in kept_exec_outs]
            chain_rewards = [float(o.reward_score) for o in kept_exec_outs
                             if o.extra_fields.get('on_answer_chain') is True]
            off_chain_rewards = [float(o.reward_score) for o in kept_exec_outs
                                 if o.extra_fields.get('on_answer_chain') is False]

            stats['exec_reward_mean'] = sum(exec_rewards) / len(exec_rewards)
            if chain_rewards:
                stats['chain_reward_mean'] = sum(chain_rewards) / len(chain_rewards)
            if off_chain_rewards:
                stats['off_chain_reward_mean'] = sum(off_chain_rewards) / len(off_chain_rewards)
            if chain_rewards and off_chain_rewards:
                stats['chain_off_chain_reward_gap'] = (
                    stats['chain_reward_mean'] - stats['off_chain_reward_mean']
                )

        # Overwrite orchestrator's dagent_stats snapshot with complete training-layer fields
        outs[0].extra_fields['dagent_stats'] = copy.deepcopy(stats)

    return outs
