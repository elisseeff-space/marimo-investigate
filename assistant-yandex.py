import marimo

__generated_with = "0.20.2"
app = marimo.App()


app._unparsable_cell(
    r"""
    import pathlib
    #import pickle

    import time, json
    from yandex-ai-studio-sdk import YCloudML
    from yandex-ai-studio-sdk.auth import APIKeyAuth
    from yandex-ai-studio-sdk.search_indexes import (
        StaticIndexChunkingStrategy,
        TextSearchIndexType,
    )

    # <путь_к_файлам_с_примерами>
    docspath = "./Documents/ya-assistant"

    in_file = "./elis_yagpt.json"
    file = open(in_file, 'r')
    econfig = json.load(file)

    sdk = YCloudML(
        folder_id = econfig["x-folder-id"],
        auth = APIKeyAuth(econfig["yandexgpt_key"]),
    )
    """,
    name="_"
)


@app.cell
def _(docspath, pathlib, sdk):
    paths = pathlib.Path(docspath).iterdir()
    files = []
    # Загрузим файлы с примерами
    # Файлы будут храниться 5 дней
    for path in paths:
        file_1 = sdk.files.upload(path, ttl_days=5, name=str(path), expiration_policy='static', labels={'author': 'Jury Moroz'})
        files.append(file_1)
    return file_1, files


@app.cell
def _(files):
    files[0]
    return


@app.cell
def _(file_1, sdk):
    sdk.files.info('fvttnss5nkj4u42luqkm')
    second = sdk.files.get(file_1.id)
    return


@app.cell
def _(sdk):
    count = 0
    for _ in sdk.files.list():
        count = count + 1  #_.delete()
    return (count,)


@app.cell
def _(count):
    count
    return


@app.cell
def _(sdk):
    files_1 = sdk.files.list()
    return (files_1,)


@app.cell
def _(files_1):
    for _ in files_1:
        print(_)
        break
    return


@app.cell
def _(sdk):
    second_1 = sdk.files.get(_.id)
    return (second_1,)


@app.cell
def _(second_1):
    second_1
    return


@app.cell
def _(StaticIndexChunkingStrategy, TextSearchIndexType, files_1, sdk):
    # Создадим индекс для полнотекстового поиска по загруженным файлам
    # Максимальный размер фрагмента — 700 токенов с перекрытием 300 токенов
    operation = sdk.search_indexes.create_deferred(files_1, index_type=TextSearchIndexType(chunking_strategy=StaticIndexChunkingStrategy(max_chunk_size_tokens=700, chunk_overlap_tokens=300)))
    return (operation,)


@app.cell
def _(operation):
    # Дождемся создания поискового индекса
    search_index = operation.wait()
    return (search_index,)


@app.cell
def _(sdk, search_index):
    # Создадим инструмент для работы с поисковым индексом.
    # Или даже с несколькими индексами, если бы их было больше.
    tool = sdk.tools.search_index(search_index)
    return (tool,)


@app.cell
def _(sdk, tool):
    # Создадим ассистента для модели YandexGPT Pro Latest
    # Он будет использовать инструмент поискового индекса
    assistant = sdk.assistants.create("yandexgpt", tools=[tool])
    return


@app.cell
def _(sdk):
    thread = sdk.threads.create()
    return


@app.cell
def _(sdk):
    input_text = ''
    assistant_1 = sdk.assistants.get('fvtgkjka1pctrn72jh5g')
    # Активируем ранее созданного ассистента
    thread_1 = sdk.threads.get('fvtil41e0rvek6t4sqpv')
    while input_text != 'exit':
        print('Введите ваш вопрос ассистенту:')
        input_text = input()
        if input_text != 'exit':
            thread_1.write(input_text)
            run = assistant_1.run(thread_1)
            result = run.wait()
            print(f'Answer: {result.text}')
            print(f'Результат 1: {result.message.author.role}')  # Отдаем модели все содержимое треда
            for _ in result.message.citations:
                print(f'Citation: {_.sources}')  # Чтобы получить результат, нужно дождаться окончания запуска  # Выводим на экран ответ  #print(f"Результат 2: {result.message.thread_id}")
    return assistant_1, thread_1


@app.cell
def _(assistant_1):
    assistant_1
    return


@app.cell
def _(thread_1):
    thread_1
    return


if __name__ == "__main__":
    app.run()
