import os
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack import Pipeline
from haystack.nodes.retriever.multimodal import MultiModalRetriever

class MultiModalSearch:
    def __init__(self, document_directory: str = "new_data"):
        '''
        Arg : 
            document_directory : Path of image folder.
        '''
        self.doc_dir = document_directory
        self.document_store = InMemoryDocumentStore(embedding_dim=512)

    def collect_all_img_path(self):
        """
        Arg:
            Directory path of images
        Return : 
            images : List of images with current path
        """
        try:
            img_ext = {".jpg",".png",".jpeg"}
            images = []
            for dir_path, folders, data in os.walk(self.doc_dir):
                for file in data:
                    if os.path.splitext(file)[1].lower() in img_ext:
                        images.append(os.path.join(dir_path,file))
            print("Total Images: ",len(images))
            return images
        except Exception as e:
            print(e)

    def create_haystack_document(self):
        """
        Arg: 
            List of images with path

        Return : 
            images : List of haystack document which contain content and content_type
        """
        try:
            images = [Document(content=f"{filename}", content_type="image")
                      for filename in self.collect_all_img_path()]
            return images
        except Exception as e:
            print(e)

    def image_retriever(self, model_name: str = "sentence-transformers/clip-ViT-B-32", query_type: str = "text"):

        """
        Arg:
            model_name : model id of hugging face
            query_type : text ,audio, image
        Return:
            haystack document retriever
        
        """
        try:
            images = self.create_haystack_document()
            self.document_store.write_documents(images)

            retriever = MultiModalRetriever(
                document_store=self.document_store,
                query_embedding_model=model_name,
                query_type=query_type,
                document_embedding_models={"image": model_name},
            )

            self.document_store.update_embeddings(retriever=retriever)
            return retriever
        except Exception as e:
            print(e)

    def image_pipeline(self):
        """
        Arg:
            image retriever
        Return : 
            pipeline object
        """

        try:
            retriever = self.image_retriever()
            pipeline = Pipeline()
            pipeline.add_node(component=retriever,
                              name="retriever_text_img",
                              inputs=["Query"])
            return pipeline
        except Exception as e:
            print(e)

    def search(self, query, top_k=3):
        """
        Arg:
            query: user inpute (text)
            top_k: k similar images
        Return:
            Sorted Result
        """
        try:
            pipeline = self.image_pipeline()
            result = pipeline.run(query=query, params={"retriever_text_img": {"top_k": top_k}})
            return sorted(result["documents"], key=lambda d: d.score, reverse=True)
        except Exception as e:
            print(e)
