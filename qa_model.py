from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class QuestionAnswering:
    def __init__(self):
        self.prompt_template = ChatPromptTemplate.from_template("""
            Answer the question, in {selected_language}, as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
            provided context just say, don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n

            Answer:
        """)
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        self.chain = self.prompt_template | self.model

    def answer(self, question: str, context, selected_language: str):
        return self.chain.invoke({
            "context": context,
            "question": question,
            "selected_language": selected_language
        }, return_exceptions=True)

