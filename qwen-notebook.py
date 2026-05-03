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
def _(file_upload, mo):
    raw = file_upload.value
    if not raw:
        file_list = []
        file_names = []
        file_count = 0
    elif hasattr(raw, 'contents') and hasattr(raw, 'name'):
        names = []
        for i in range(100):
            n = raw.name(i)
            if n and n.strip():
                names.append(n)
            else:
                break
        file_list = raw
        file_names = names
        file_count = len(names)
    else:
        file_list = list(raw) if hasattr(raw, '__iter__') else [raw]
        file_count = len(file_list)
        file_names = [getattr(f, 'name', str(f)) for f in file_list]
    add_button = mo.ui.run_button(label="Add Files to List")
    current_files, current_names, current_count = file_list, file_names, file_count
    add_button, current_files, current_names, current_count
    return add_button, current_count, current_files, current_names


@app.cell
def _():
    uploaded_files = []
    uploaded_names = []
    return uploaded_files, uploaded_names


@app.cell
def _(
    add_button,
    current_count,
    current_files,
    current_names,
    mo,
    uploaded_files,
    uploaded_names,
):
    if add_button.value and current_count > 0:
        for f, name in zip(current_files, current_names):
            if name not in uploaded_names:
                uploaded_files.append(f)
                uploaded_names.append(name)

    display = mo.md(f"**{len(uploaded_files)}** file{'s' if len(uploaded_files) != 1 else ''} in list:\n\n" + "\n".join(f"- {n}" for n in uploaded_names))
    display
    return


@app.cell
def _(
    Path,
    current_files,
    current_names,
    read_document,
    tempfile,
    uploaded_files,
    uploaded_names,
):
    import base64
    def extract_text_from_files(files):
        """Extract text from uploaded files."""
        if not files or len(files) == 0:
            return ""

        extracted = []
        try:
            file_count = len(files)
        except TypeError:
            file_count = 1

        has_api = hasattr(files, 'name') and callable(getattr(files, 'name', None))

        for i in range(file_count):
            try:
                if has_api:
                    fname = files.name(i)
                    fcontent_raw = files.contents(i)
                elif isinstance(files[i], (list, tuple)):
                    fname = files[i][0]
                    fcontent_raw = files[i][1]
                elif hasattr(files[i], 'contents'):
                    fname = getattr(files[i], 'name', f'file_{i}')
                    fcontent_raw = files[i].contents
                else:
                    print(f"Unknown file format: {type(files[i])}")
                    continue

                fcontent = base64.b64decode(fcontent_raw) if isinstance(fcontent_raw, str) else fcontent_raw

                suffix = Path(fname).suffix.lower()
                with tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False
                ) as tmp:
                    tmp.write(fcontent)
                    tmp_path = Path(tmp.name)

                if suffix == ".txt":
                    text = fcontent.decode("utf-8", errors="replace")
                else:
                    text = read_document(tmp_path) or ""

                if text:
                    extracted.append(f"## {fname}\n\n{text}")
            except Exception as e:
                print(f"Error processing file at index {i}: {e}")
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return "\n\n---\n\n".join(extracted)

    combined_files = uploaded_files + current_files
    combined_names = list(set(uploaded_names + current_names))
    document_text = extract_text_from_files(combined_files) if combined_files else ""
    text_preview = document_text[:500] + ("..." if len(document_text) > 500 else "")
    text_len = len(document_text)
    text_preview, text_len
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
                f"{document_text}\n"
                "---\n"
            )

            def run_query():
                llm = LLMManager()
                return llm.query(
                    user_message,
                    system_prompt=system_prompt,
                    response_model=None,
                    use_cache=False,
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
