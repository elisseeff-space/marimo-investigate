import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import tempfile
    from pathlib import Path
    from common.document_reader import read_document
    from adapters.llm import LLMManager
    import concurrent.futures

    return LLMManager, Path, concurrent, mo, read_document, tempfile


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Document Chat

    Upload documents and chat with an AI model about their contents.
    """)
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".pdf", ".docx", ".txt"],
        multiple=True,
        label="Upload Documents",
    )
    file_upload
    return (file_upload,)


@app.cell
def _(file_upload):
    file_count = len(file_upload.value) if file_upload.value else 0
    file_count
    return (file_count,)


@app.cell
def _(file_count, mo):
    if file_count > 0:
        mo.md(f"**{file_count}** file{'s' if file_count != 1 else ''} uploaded")
    else:
        mo.md("No files uploaded yet")
    return


@app.cell
def _(Path, file_upload, read_document, tempfile):
    def extract_text_from_files(files):
        """Extract text from uploaded files."""
        if not files:
            return ""

        extracted = []
        for file_data in files:
            filename = file_data.name
            content = file_data.contents

            suffix = Path(filename).suffix.lower()
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)

                if suffix == ".txt":
                    text = content.decode("utf-8", errors="replace")
                else:
                    text = read_document(tmp_path) or ""

                if text:
                    extracted.append(f"## {filename}\n\n{text}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return "\n\n---\n\n".join(extracted)

    document_text = extract_text_from_files(file_upload.value) if file_upload.value else ""
    document_text[:500] + ("..." if len(document_text) > 500 else "")
    return (document_text,)


@app.cell
def _(document_text, mo):
    if document_text:
        mo.md(f"""
        <details>
        <summary>View extracted text</summary>
        <pre>{document_text[:2000]}</pre>
        </details>
        """)
    else:
        mo.md("_Upload documents to see extracted text here_")
    return


@app.cell
def _(mo):
    question = mo.ui.text(
        label="Ask a question",
        value="give the main theses",
        placeholder="Type your question here..."
    )
    ask_button = mo.ui.run_button(label="Ask")
    mo.vstack([question, ask_button])
    return ask_button, question


@app.cell
def _(LLMManager, ask_button, concurrent, document_text, mo, question):
    if not ask_button.value:
        answer_display = mo.md("_Click 'Ask' to get an answer_")
    elif not document_text:
        answer_display = mo.md("Please upload documents first so I can answer questions about them.")
    else:
        user_message = question.value or ""
        if not user_message.strip():
            answer_display = mo.md("Please type a question first.")
        else:
            system_prompt = (
                "You are a helpful assistant analyzing uploaded documents.\n"
                "Use the following document content to answer questions accurately.\n"
                "If the answer is not in the documents, say so clearly.\n\n"
                "---\n"
                f"{document_text[:8000]}\n"
                "---\n"
            )

            def run_query():
                llm = LLMManager()
                return llm.query(
                    user_message,
                    system_prompt=system_prompt,
                    response_model=None,
                )

            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_query)
                    result = future.result()
                if hasattr(result, "content"):
                    answer_text = result.content
                else:
                    answer_text = str(result)
                answer_display = mo.md(f"**Answer:**\n\n{answer_text}")
            except Exception as e:
                answer_display = mo.md(f"Error: {str(e)}")

    answer_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    **Tips:**
    - Upload PDF, DOCX, or TXT files
    - Ask specific questions about document content
    - Multiple files are combined into one context
    """)
    return


if __name__ == "__main__":
    app.run()
