"""Pydantic data models for the reasoning tree/DAG."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
    """A single atomic step in one trajectory."""

    id: str
    trajectory_id: str
    step_index: int
    raw_text: str
    canonical_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    is_conclusion: bool = False  # True if this step is the trajectory's explicit answer


class ReasoningNode(BaseModel):
    """A merged equivalence class of steps that appears in the DAG."""

    node_id: str
    canonical_text: str
    supporting_steps: List[str] = Field(default_factory=list)
    trajectory_ids: List[str] = Field(default_factory=list)
    support_count: int = 0
    representative_raw_text: Optional[str] = None
    virtual: bool = False  # True for synthetic START/END super-nodes
    is_conclusion: bool = False  # True if any supporting step is a conclusion
    conclusion_trajectories: List[str] = Field(default_factory=list)


class ReasoningEdge(BaseModel):
    """A directed transition between two semantic clusters in the DAG."""

    source: str
    target: str
    trajectories: List[str] = Field(default_factory=list)
    count: int = 0
    probability: Optional[float] = None  # count / sum(out_edges from source)
    virtual: bool = False  # True for edges to/from a virtual START/END node


class TrajectoryInput(BaseModel):
    """Input record describing one trajectory."""

    trajectory_id: str
    text: str
    conclusion_text: Optional[str] = None  # explicit reason/answer (e.g. principal_reason)


class ResponsesPayload(BaseModel):
    """Top-level input format for the CLI."""

    prompt: Optional[str] = None
    responses: List[TrajectoryInput]


class GraphDump(BaseModel):
    """Serialization-friendly view of the built DAG."""

    prompt: Optional[str] = None
    trajectories: List[str]
    nodes: List[ReasoningNode]
    edges: List[ReasoningEdge]
    metrics: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
