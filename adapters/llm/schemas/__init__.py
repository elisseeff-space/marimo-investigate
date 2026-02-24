"""
Pydantic schemas for LLM structured output.

Submodules:
- government_docs: Russian government document extraction schemas
"""

# Re-export from original schemas.py for backwards compatibility
from adapters.llm.schemas_original import (
    EXTRACTION_SCHEMAS,
    DERIVATION_SCHEMAS,
    BusinessConceptItem,
    BusinessConceptResponse,
    BusinessConceptMultiFileResult,
    BusinessConceptMultiResponse,
    TypeDefinitionItem,
    TypeDefinitionResponse,
    TechnologyItem,
    TechnologyResponse,
    ExternalDependencyItem,
    ExternalDependencyResponse,
    TestItem,
    TestResponse,
    MethodItem,
    MethodResponse,
    DirectoryClassificationItem,
    DirectoryClassificationResponse,
    DerivedElementItem,
    DerivedElementResponse,
    DerivedRelationshipItem,
    DerivedRelationshipResponse,
    get_extraction_schema,
    get_derivation_schema,
)

# Government documents schemas
from .government_docs import (
    DocumentInfo,
    FederalProject,
    FederalProjectIndicator,
    GovernmentDocumentExtraction,
    ImplementationPeriod,
    Indicator,
    NationalGoal,
    NationalProject,
    Relationship,
    StateProgram,
    TargetValues,
)

__all__ = [
    # Original extraction schemas (backwards compatibility)
    "EXTRACTION_SCHEMAS",
    "DERIVATION_SCHEMAS",
    "BusinessConceptItem",
    "BusinessConceptResponse",
    "BusinessConceptMultiFileResult",
    "BusinessConceptMultiResponse",
    "TypeDefinitionItem",
    "TypeDefinitionResponse",
    "TechnologyItem",
    "TechnologyResponse",
    "ExternalDependencyItem",
    "ExternalDependencyResponse",
    "TestItem",
    "TestResponse",
    "MethodItem",
    "MethodResponse",
    "DirectoryClassificationItem",
    "DirectoryClassificationResponse",
    "DerivedElementItem",
    "DerivedElementResponse",
    "DerivedRelationshipItem",
    "DerivedRelationshipResponse",
    "get_extraction_schema",
    "get_derivation_schema",
    # Government documents
    "GovernmentDocumentExtraction",
    "DocumentInfo",
    "NationalGoal",
    "NationalProject",
    "FederalProject",
    "StateProgram",
    "Indicator",
    "FederalProjectIndicator",
    "TargetValues",
    "ImplementationPeriod",
    "Relationship",
]
