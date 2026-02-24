"""
Pydantic schemas for Russian government document extraction.

Generated from json-schema.json for extracting national development goals,
projects, and indicators from Russian government documents.

These models exactly match the JSON schema structure for proper serialization.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Document Info
# =============================================================================


class DocumentInfo(BaseModel):
    """Metadata about the source document."""

    model_config = ConfigDict(populate_by_name=True)

    title: str = Field(description="Full title of the document")
    document_type: Literal[
        "Указ",
        "Постановление",
        "Паспорт НП",
        "Единый план",
        "Стратегия",
        "Госпрограмма",
        "Другое",
    ] = Field(description="Type of the document")
    date: str | None = Field(default=None, description="Document date (YYYY-MM-DD format)")
    number: str | None = Field(default=None, description="Document number (e.g., '309', '1710')")
    source_file: str = Field(description="Original filename")


# =============================================================================
# Indicators
# =============================================================================


class TargetValues(BaseModel):
    """Target values by year."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    year_2024: float | None = Field(default=None, alias="2024")
    year_2025: float | None = Field(default=None, alias="2025")
    year_2026: float | None = Field(default=None, alias="2026")
    year_2027: float | None = Field(default=None, alias="2027")
    year_2028: float | None = Field(default=None, alias="2028")
    year_2029: float | None = Field(default=None, alias="2029")
    year_2030: float | None = Field(default=None, alias="2030")
    year_2036: float | None = Field(default=None, alias="2036")


class Indicator(BaseModel):
    """Indicator/metric for measuring goal or project achievement."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Indicator identifier (e.g., 'НЦР-1.1')")
    name: str = Field(description="Indicator name/description")
    unit: str | None = Field(default=None, description="Unit of measurement")
    baseline_year: int | None = Field(default=None, description="Baseline year for measurement")
    baseline_value: float | None = Field(default=None, description="Value at baseline year")
    target_values: TargetValues | None = Field(default=None, description="Target values by year")
    data_source: str | None = Field(default=None, description="Information system or source for data")
    is_quasi_indicator: bool | None = Field(default=None, description="True if this is a quasi-indicator")


class FederalProjectIndicator(BaseModel):
    """Indicator for a federal project."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Federal project indicator ID")
    name: str = Field(description="Federal project indicator name")
    unit: str | None = Field(default=None, description="Unit of measurement")
    target_values: TargetValues | None = Field(default=None, description="Target values by year")


# =============================================================================
# Federal Projects
# =============================================================================


class FederalProject(BaseModel):
    """Federal project within a national project."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Federal project ID (e.g., 'ФП-1')")
    name: str = Field(description="Federal project name")
    description: str | None = Field(default=None, description="Federal project description")
    goal: str | None = Field(default=None, description="Federal project goal")
    indicators: list[FederalProjectIndicator] = Field(default_factory=list)


# =============================================================================
# National Goals
# =============================================================================


class NationalGoal(BaseModel):
    """National Development Goal (НЦР)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Unique identifier (e.g., 'НЦР-1', 'НЦР-а')")
    name: str = Field(description="Full name of the national goal")
    description: str | None = Field(default=None, description="Detailed description of the goal")
    curator: str | None = Field(default=None, description="Curator of the goal (government official)")
    indicators: list[Indicator] = Field(default_factory=list)
    factors_and_tools: list[str] = Field(default_factory=list)


# =============================================================================
# National Projects
# =============================================================================


class ImplementationPeriod(BaseModel):
    """Implementation period for a national project."""

    model_config = ConfigDict(populate_by_name=True)

    start_date: str | None = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="End date (YYYY-MM-DD)")


class NationalProject(BaseModel):
    """National Project (НП)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(description="Unique identifier (e.g., 'НП-1')")
    name: str = Field(description="Full name of the national project")
    short_name: str | None = Field(default=None, description="Short name for diagrams")
    description: str | None = Field(default=None, description="Description of the national project")
    implementation_period: ImplementationPeriod | None = Field(default=None)
    curator: str | None = Field(default=None, description="Project curator (Deputy Prime Minister)")
    head: str | None = Field(default=None, description="Project head (Minister)")
    administrator: str | None = Field(default=None, description="Project administrator")
    target_groups: list[str] = Field(default_factory=list)
    related_national_goals: list[str] = Field(default_factory=list)
    indicators: list[Indicator] = Field(default_factory=list)
    federal_projects: list[FederalProject] = Field(default_factory=list)


# =============================================================================
# State Programs
# =============================================================================


class StateProgram(BaseModel):
    """State Program (ГП)."""

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None, description="State program ID")
    name: str = Field(description="State program name")
    description: str | None = Field(default=None, description="State program description")
    related_national_projects: list[str] = Field(default_factory=list)


# =============================================================================
# Relationships
# =============================================================================


class Relationship(BaseModel):
    """Relationship between extracted elements for ArchiMate graph."""

    model_config = ConfigDict(populate_by_name=True)

    source_id: str = Field(description="Source element ID")
    target_id: str = Field(description="Target element ID")
    relationship_type: Literal[
        "Aggregation",
        "Composition",
        "Association",
        "Realization",
        "Influence",
        "Serving",
    ] = Field(description="ArchiMate relationship type")
    name: str | None = Field(default=None, description="Relationship name/description")


# =============================================================================
# Main Response Schema
# =============================================================================


class GovernmentDocumentExtraction(BaseModel):
    """
    Extracted national development goals, projects, and indicators
    from Russian government documents.
    
    This model exactly matches json-schema.json structure.
    """

    model_config = ConfigDict(populate_by_name=True)

    document_info: DocumentInfo = Field(description="Metadata about the source document")
    national_goals: list[NationalGoal] = Field(
        default_factory=list,
        description="National Development Goals (НЦР) defined in the document",
    )
    national_projects: list[NationalProject] = Field(
        default_factory=list,
        description="National Projects (НП) aimed at achieving national goals",
    )
    state_programs: list[StateProgram] = Field(
        default_factory=list,
        description="State Programs (ГП) mentioned in the document",
    )
    relationships: list[Relationship] = Field(
        default_factory=list,
        description="Relationships between extracted elements for ArchiMate graph",
    )

    def to_json_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary matching exact JSON schema structure.
        
        Handles special field names like "2024", "2025" etc. in target_values.
        """
        return self.model_dump(by_alias=True, exclude_none=False, mode="json")
