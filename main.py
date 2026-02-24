"""
Main entry point demonstrating LLM structured output.

The LLM is configured to always return structured responses according to
the JSON schema defined in json-schema.json (GovernmentDocumentExtraction).

Usage:
    uv run main.py                          # Run with example prompt
    uv run main.py --input-doc <file>       # Process specific file
    uv run main.py -i <file>                # Short form
    uv run main.py -i <file> -o result.json # Save result to JSON file
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from adapters.llm import LLMManager
from adapters.llm.schemas import GovernmentDocumentExtraction
from common.document_reader import read_document


def read_file_content(file_path: str) -> str:
    """
    Read content from a file.

    For PDF and DOCX files, uses proper document extraction.
    For text files, reads as plain text.

    Args:
        file_path: Path to the file to read

    Returns:
        File content as string

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file can't be read
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Use document reader for PDF and DOCX files
    suffix = path.suffix.lower()
    if suffix in (".pdf", ".docx"):
        content = read_document(path)
        if content is not None:
            return content
        # Fall through to plain text read if document reader fails

    # Try to read as text
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with different encoding
        return path.read_text(encoding="latin-1")


def load_prompt_from_file(prompt_value: str) -> str:
    """
    Load prompt from file if the value is a file path, otherwise return as-is.

    Args:
        prompt_value: Either a file path or direct prompt text

    Returns:
        Prompt content (loaded from file or original text)
    """
    path = Path(prompt_value)
    if path.exists() and path.is_file():
        return read_file_content(prompt_value)
    return prompt_value


def create_extraction_prompt(content: str, filename: str) -> str:
    """
    Create a prompt for extracting government document data.
    
    Args:
        content: The document content
        filename: Original filename for reference
        
    Returns:
        Formatted prompt for the LLM
    """
    return f"""
Extract information from the following Russian government document.

Source file: {filename}

Document content:
---
{content}
---

Extract all national goals (НЦР), national projects (НП), state programs (ГП), 
and their relationships. Return the data in the structured format according to 
the GovernmentDocumentExtraction schema.

Important:
- Preserve original Russian names and terminology
- Extract all indicators with their target values
- Include document metadata (title, type, date, number)
- Identify relationships between goals, projects, and programs
"""


def export_to_json(
    result: GovernmentDocumentExtraction,
    output_path: str,
    input_file: str | None = None,
    include_metadata: bool = True,
) -> str:
    """
    Export extraction result to JSON file matching the exact schema structure.
    
    Args:
        result: The extraction result
        output_path: Path to output JSON file
        input_file: Optional input file path for metadata
        include_metadata: If True, adds _export_info; if False, outputs pure schema
        
    Returns:
        Path to created JSON file
    """
    # Convert to dict matching JSON schema
    data = result.to_json_dict()
    
    if include_metadata:
        # Add export metadata
        export_metadata = {
            "_export_info": {
                "exported_at": datetime.now().isoformat(),
                "schema_version": "1.0",
                "source_file": input_file,
            }
        }
        # Merge with data (metadata at top level for auditing)
        output_data = {**export_metadata, **data}
    else:
        # Pure schema output
        output_data = data
    
    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return str(output_file)


def generate_output_path(input_file: str | None, output_dir: str = "output") -> str:
    """
    Generate output JSON file path based on input file.
    
    Args:
        input_file: Input file path (if any)
        output_dir: Output directory
        
    Returns:
        Generated output file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_file:
        # Use input filename with .json extension
        input_name = Path(input_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(output_path / f"{input_name}_{timestamp}.json")
    else:
        # Default name for example prompt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(output_path / f"extraction_{timestamp}.json")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Extract structured data from Russian government documents using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run main.py
    uv run main.py --input-doc documents/ukaz_309.pdf
    uv run main.py -i path/to/document.txt
    uv run main.py -i document.txt -o result.json
    uv run main.py -i document.txt --output-json  # Auto-generate output path
    uv run main.py --user-prompt "Extract key information"  # Custom user prompt (text)
    uv run main.py --user-prompt prompt.txt  # Load user prompt from .txt file
    uv run main.py --user-prompt prompt.md  # Load user prompt from .md file
    uv run main.py --system-prompt system.md --user-prompt user.md  # Load both from Markdown files
        """,
    )

    parser.add_argument(
        "-i", "--input-doc",
        type=str,
        help="Path to the document file to process",
    )

    parser.add_argument(
        "-o", "--output-json",
        type=str,
        nargs="?",
        const="auto",
        metavar="FILE",
        help="Export result to JSON file (auto-generates path if FILE not specified)",
    )

    parser.add_argument(
        "--pure-json",
        action="store_true",
        help="Output pure JSON matching schema exactly (no _export_info metadata)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for auto-generated JSON output (default: output)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (force fresh LLM call)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt (file path: *.txt, *.md or direct text)",
    )

    parser.add_argument(
        "--user-prompt",
        type=str,
        help="Custom user prompt (file path: *.txt, *.md or direct text)",
    )

    args = parser.parse_args()
    
    print("Hello from marimo-investigate!")
    print("=" * 60)
    
    # Create LLM Manager with default structured output schema
    llm = LLMManager(default_response_model=GovernmentDocumentExtraction)

    print(f"LLM Provider: {llm.provider_name}")
    print(f"Model: {llm.model}")
    print(f"Default response model: GovernmentDocumentExtraction")
    print("=" * 60)

    # Determine input source and build prompt
    input_file = args.input_doc
    
    # Load prompts from files if paths are provided
    system_prompt = load_prompt_from_file(args.system_prompt) if args.system_prompt else None
    user_prompt = load_prompt_from_file(args.user_prompt) if args.user_prompt else None

    # If custom prompts are provided, use them
    if user_prompt:
        # Custom user prompt provided (loaded from file or direct text)
        prompt = user_prompt
        prompt_source = "file" if args.user_prompt and Path(args.user_prompt).exists() else "text"
        print(f"\nUsing custom user prompt ({len(prompt)} characters) from {prompt_source}")
    elif input_file:
        # Read from file and create extraction prompt
        print(f"\nReading document: {args.input_doc}")
        try:
            content = read_file_content(args.input_doc)
            filename = Path(args.input_doc).name
            prompt = create_extraction_prompt(content, filename)
            print(f"Document loaded ({len(content)} characters)")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
        except IOError as e:
            print(f"Error reading file: {e}")
            return
    else:
        # Use example prompt
        filename = "example_prompt.txt"
        prompt = """
        Extract information from the following text about Russian national development goals and projects.

        Text: "Указ Президента Российской Федерации от 7 мая 2024 года № 309
        'О национальных целях развития Российской Федерации на период до 2030 года
        и на перспективу до 2036 года'.

        Национальные цели:
        1. НЦР-1: Сбережение народа России
        2. НЦР-2: Возможности для самореализации и развития талантов
        3. НЦР-3: Искусственный интеллект и цифровая трансформация

        Национальные проекты:
        - НП-1: Здоровье
        - НП-2: Образование
        - НП-3: Цифровая экономика
        """
        print("\nUsing example prompt (no input file specified)")

    print("-" * 60)
    print("Sending query to LLM...")

    # Query with optional cache override and custom prompts
    use_cache = not args.no_cache
    response = llm.query(prompt, use_cache=use_cache, system_prompt=system_prompt)
    
    # Handle failed response
    if hasattr(response, 'response_type') and response.response_type == "failed":
        print(f"\nError: {response.error}")
        print(f"Error type: {response.error_type}")
        return
    
    # Display results
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    
    # Document Info
    if response.document_info:
        print("\n📄 Document Info:")
        doc = response.document_info
        print(f"   Title: {doc.title}")
        print(f"   Type: {doc.document_type}")
        if doc.date:
            print(f"   Date: {doc.date}")
        if doc.number:
            print(f"   Number: {doc.number}")
        print(f"   Source: {doc.source_file}")
    
    # National Goals
    if response.national_goals:
        print(f"\n🎯 National Goals ({len(response.national_goals)}):")
        for goal in response.national_goals:
            print(f"   • {goal.id}: {goal.name}")
            if goal.description and args.verbose:
                print(f"      Description: {goal.description}")
            if goal.curator and args.verbose:
                print(f"      Curator: {goal.curator}")
            if goal.indicators:
                print(f"      Indicators: {len(goal.indicators)}")
    
    # National Projects
    if response.national_projects:
        print(f"\n📋 National Projects ({len(response.national_projects)}):")
        for project in response.national_projects:
            print(f"   • {project.id}: {project.name}")
            if project.short_name and args.verbose:
                print(f"      Short name: {project.short_name}")
            if project.description and args.verbose:
                print(f"      Description: {project.description}")
            if project.related_national_goals:
                print(f"      Related goals: {', '.join(project.related_national_goals)}")
            if project.federal_projects:
                print(f"      Federal projects: {len(project.federal_projects)}")
    
    # State Programs
    if response.state_programs:
        print(f"\n🏛️ State Programs ({len(response.state_programs)}):")
        for program in response.state_programs:
            print(f"   • {program.id or 'N/A'}: {program.name}")
    
    # Relationships
    if response.relationships:
        print(f"\n🔗 Relationships ({len(response.relationships)}):")
        for rel in response.relationships:
            print(f"   • {rel.source_id} → {rel.target_id} ({rel.relationship_type})")
    
    # Export to JSON if requested
    if args.output_json:
        print("\n" + "=" * 60)
        print("EXPORTING TO JSON")
        print("=" * 60)

        # Determine output path
        if args.output_json == "auto":
            output_path = generate_output_path(input_file, args.output_dir)
        else:
            output_path = args.output_json
        
        # Export with option for pure JSON (no metadata)
        include_metadata = not args.pure_json
        try:
            exported_file = export_to_json(response, output_path, input_file, include_metadata)
            print(f"✓ JSON exported to: {exported_file}")
            
            # Validate against schema
            print("\nValidating JSON structure...")
            with open(exported_file, "r", encoding="utf-8") as f:
                exported_data = json.load(f)
            
            # Check required fields
            required_fields = ["document_info", "national_goals", "national_projects"]
            missing = [f for f in required_fields if f not in exported_data]
            if missing:
                print(f"⚠ Warning: Missing required fields: {missing}")
            else:
                print("✓ All required fields present")
                
        except Exception as e:
            print(f"✗ Export failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Structured extraction complete!")
    
    # Cache info
    if hasattr(response, 'response_type'):
        cache_status = "CACHED" if response.response_type == "cached" else "LIVE"
        print(f"   Response source: {cache_status}")


if __name__ == "__main__":
    main()
