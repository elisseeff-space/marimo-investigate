import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os
    import openai
    import marimo as mo

    from dotenv import load_dotenv

    load_dotenv()
    folder_id=os.environ.get("YC_FOLDER_ID")
    auth=os.environ.get("YC_APIKEY")
    model=os.environ.get("YANDEX_MODEL")

    print("hello", folder_id, auth)

    client = openai.OpenAI(
        api_key=auth,
        base_url="https://ai.api.cloud.yandex.net/v1",
        project= folder_id
    )

    response = client.responses.create(
        model=f"gpt://{folder_id}/{model}",
        input="Придумай 3 необычные идеи для стартапа в сфере путешествий.",
        temperature=0.8,
        max_output_tokens=1500
    )

    print(response.output[0].content[0].text)
    return auth, folder_id


@app.cell
def _():
    return auth, folder_id


@app.cell
def _():
    import yandex_ai_studio_sdk

    assistants = yandex_ai_studio_sdk.Assistants(
        auth=auth,
        folder_id=folder_id
    )

    all_assistants = list(assistants.list())
    for a in all_assistants:
        print(f"ID: {a.id}, Name: {a.name}, Model: {a.model}")
    return


if __name__ == "__main__":
    app.run()
