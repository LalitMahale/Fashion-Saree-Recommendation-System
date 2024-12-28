import os
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack import Pipeline
from haystack.nodes.retriever.multimodal import MultiModalRetriever

class MultiModalSearch:
    def __init__(self):
        self.document_store = InMemoryDocumentStore(
            embedding_dim=512
        )

        doc_dir = "new_data"

        # Assuming you have images in the directory "new_data"
        images = [Document(content=f"{doc_dir}/{filename}", content_type="image")
                  for filename in os.listdir(f"./{doc_dir}")]

        # Write the documents to the document store
        self.document_store.write_documents(images)

        # Initialize the multimodal retriever
        self.retriever_text_to_image = MultiModalRetriever(
            document_store=self.document_store,
            query_embedding_model="sentence-transformers/clip-ViT-B-32",
            query_type="text",
            document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
        )

        # Update the document embeddings
        self.document_store.update_embeddings(retriever=self.retriever_text_to_image)

        # Initialize the pipeline
        self.pipeline = Pipeline()

        # Add the retriever as the first node in the pipeline
        self.pipeline.add_node(component=self.retriever_text_to_image,
                               name="retriever_text_img",
                               inputs=["Query"])

    def search(self, query, top_k=3):
        # Run the pipeline with the query
        result = self.pipeline.run(query=query, params={"retriever_text_img": {"top_k": top_k}})

        # Return sorted documents based on their relevance score
        return sorted(result["documents"], key=lambda d: d.score, reverse=True)

