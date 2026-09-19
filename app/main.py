import argparse

from app.agent.financial_agent import create_financial_agent
from app.ingestion.pdf_ingestion import ingest_pdf


def cmd_ingest(file_path: str):
    result = ingest_pdf(file_path)
    print("Ingested document:")
    for key, value in result.items():
        print(f"  {key}: {value}")


def cmd_ask(question: str):
    agent = create_financial_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print(result["messages"][-1].content)


def main():
    parser = argparse.ArgumentParser(
        description="Financial report RAG agent."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a PDF into the knowledge base."
    )
    ingest_parser.add_argument("file_path", help="Path to the PDF file.")

    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question about the most recently ingested PDF."
    )
    ask_parser.add_argument("question", help="The question to ask.")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args.file_path)
    elif args.command == "ask":
        cmd_ask(args.question)


# def main(argv=None):
#     parser = argparse.ArgumentParser(
#         description="Financial report RAG agent."
#     )
#
#     subparsers = parser.add_subparsers(
#         dest="command",
#         required=True
#     )
#
#     ingest_parser = subparsers.add_parser("ingest")
#     ingest_parser.add_argument("file_path")
#
#     ask_parser = subparsers.add_parser("ask")
#     ask_parser.add_argument("question")
#
#     args = parser.parse_args(argv)
#
#     if args.command == "ingest":
#         cmd_ingest(args.file_path)
#
#     elif args.command == "ask":
#         cmd_ask(args.question)

if __name__ == "__main__":
    # main(['ingest', 'data/uploads/vinamilk_2023_consolidated_financial_statements.pdf'])
    # main(['ask', "What was Vinamilk's net revenue in 2023?"])
    main()