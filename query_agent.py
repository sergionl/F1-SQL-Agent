"""Agente SQL conversacional sobre la base de datos de F1 (LangChain + OpenAI).

Requiere la variable de entorno OPENAI_API_KEY (o la solicita de forma interactiva).

Uso:
    python query_agent.py --question "En que equipo corre Charles Leclerc"
    python query_agent.py --interactive
    python query_agent.py --db agente_sql_f1.db --interactive --quiet
"""

import argparse
import os
from getpass import getpass

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI


def ensure_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("Ingresa tu OpenAI API Key: ")


def build_agent(db_path: str, model: str = "gpt-3.5-turbo", verbose: bool = True):
    db = SQLDatabase.from_uri(f'sqlite:///{db_path}')
    llm = ChatOpenAI(model=model, temperature=0)
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=verbose,
    )
    return agent_executor


def ask(agent_executor, question: str) -> str:
    respuesta = agent_executor.invoke({"input": question})
    return respuesta['output']


def interactive_loop(agent_executor) -> None:
    print("Modo interactivo. Escribe 'salir' o 'exit' para terminar.\n")
    while True:
        try:
            question = input("Pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ('salir', 'exit', 'quit'):
            break
        print(ask(agent_executor, question))
        print()


def main():
    parser = argparse.ArgumentParser(description="Consulta la base de datos de F1 en lenguaje natural")
    parser.add_argument('--db', default='agente_sql_f1.db', help='Ruta de la base de datos SQLite (default: agente_sql_f1.db)')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='Modelo de OpenAI a usar (default: gpt-3.5-turbo)')
    parser.add_argument('--question', help='Pregunta puntual a responder (modo no interactivo)')
    parser.add_argument('--interactive', action='store_true', help='Inicia un loop de preguntas por consola')
    parser.add_argument('--quiet', action='store_true', help='Oculta el razonamiento intermedio del agente')
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.error('Debes pasar --question "..." o usar --interactive')

    ensure_api_key()
    agent_executor = build_agent(args.db, model=args.model, verbose=not args.quiet)

    if args.question:
        print(ask(agent_executor, args.question))

    if args.interactive:
        interactive_loop(agent_executor)


if __name__ == '__main__':
    main()
