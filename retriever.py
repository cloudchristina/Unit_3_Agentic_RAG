from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import datasets

#----------------------2. Define the Retriever Tool----------------------#
# The name and description help the agent understand when and how to use this tool
# The inputs define what parameters the tool expects (in this case, a search query)
# We’re using a BM25Retriever, which is a powerful text retrieval algorithm that doesn’t require embeddings
# The forward method processes the query and returns the most relevant guest information
class GuestInfoRetrieverTool(Tool):
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)


    def forward(self, query: str):
        results = self.retriever.invoke(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."


#----------------------1. Load Dataset, Prepare and Create Tool----------------------#
    # Load the dataset
    # Convert each guest entry into a Document object with formatted content
    # Store the Document objects in a list
def load_guest_dataset():

    guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

    # Convert dataset entries into Document objects
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]

    # Return the tool
    return GuestInfoRetrieverTool(docs)