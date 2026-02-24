import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from adapters.llm import LLMManager

    # Uses provider from .env (LLM_PROVIDER, LLM_MODEL, etc.)
    llm = LLMManager()

    # Simple query
    response = llm.query("What is Python?")
    if response.response_type == "live":
        print(response.content)
    return


@app.cell
def _():
    import pandas as pd
    df = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["New York", "London", "Paris"]
    })
    df
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import requests

    # Example using a public API to demonstrate requests usage
    response = requests.get("https://jsonplaceholder.typicode.com/posts/1")

    if response.status_code == 200:
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())
    else:
        print("Error:", response.status_code)
    return


@app.cell
def _():
    # Install required packages if not already installed (Official Python driver)
    #!pip install neo4j --quiet

    #import marimo as mo 
    from pydantic import SecretStr # For type hinting credentials safely in examples, though often just passed directly for simple demos.

    return


@app.cell
def _(mo):
    #import marimo as mo 
    from neo4j import GraphDatabase # Official driver for Neo4j connection and queries


    class MovieApp:  # Example class to demonstrate graph traversal concepts often used in Cypher tutorials

        def __init__(self, uri):
            self.driver = None


    mo.md("""
    To use **Neo4j** (a popular open-source NoSQL database) locally:
    1. Install Neo4j Desktop or Docker.
    2. Run the server on `bolt://localhost:7474` and/or http endpoint.

    Below is an example using Python to connect securely with authentication credentials stored in variables for demonstration purposes:

    $$ \text{GraphDB} = f(\{\}) $$
    """)
    return


@app.cell
def _(e):
    # Assuming a local Neo4j instance running on standard ports (bolt:7687/http:14747)
    import marimo as mo

    try:
        from neo4j import GraphDatabase

        # Replace with your actual password and URI if different defaults are used in Docker/Neo4j Desktop settings.
        username = "neo4j"
        uri = "http://localhost:17478"  # Neo4j Browser URL (HTTP) or bolt endpoint
        driver = None

    except Exception: e
    return (mo,)


app._unparsable_cell(
    """
    # Example: Connecting to local instance and executing Cypher queries

    def run_cypher_query(uri):
        \"\"\"
        Function demonstrating how to connect securely using credentials.

        In practice, never hardcode passwords. Use environment variables 
        with a .env file in your project directory for secrets management:
            export NEO4J_URI=\"bolt://localhost:7687\"

    $$ \\text{Credentials} = f(\\{\\}) $$
    \"\"\"
    from neo4j import GraphDatabase
    import os

    # Safely retrieve credentials from the environment (best practice)
    neo_uri_str = \"http://localhost:17478\" # Neo4j Browser URL or bolt endpoint for newer versions like 5+
    username_env_var_name = \"NEO4J_USERNAME\"
    password_env_var_name = \"PASSWORD\"

    try:
        username_val = os.getenv(username_env_var_name, \"\")

    except Exception as e:

    mo.md(\"\"\"
    ### Using `requests` with an older version of Neo4j (v3/v2)

    If you have a legacy installation and prefer using the standard HTTP API 
    (without installing Python drivers), here is how to connect via requests.

    $$ \\text{API} = f(\\{\\}) $$
    \"\"\")
    """,
    name="_"
)


app._unparsable_cell(
    """
    # Example connection to Neo4j (Standard approach)
    import marimo as mo 

    try:
        from neo4j import GraphDatabase

    except ImportError:

    mo.md(\"\"\"
    ### Querying Data with Cypher

    After establishing the driver, you can execute **Cypher** queries. 
    This is similar to SQL but designed for graph traversal.

    $$ \\text{MATCH} = f(\\{\\}) $$
    \"\"\")
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
