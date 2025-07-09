import os, os.path
import chromadb
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_chroma import Chroma
import datetime
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages, AnyMessage
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import base64
from PIL import Image
from io import BytesIO

chosen_emb_model = "CLIP"
# chosen_emb_model = "MXBAI"

match chosen_emb_model:
    case "CLIP":
        dir = "../../brain_DB_CLIP"
        emb_func = OpenCLIPEmbeddings(model_name = 'ViT-B-32', checkpoint = 'laion2b_e16') #Multimodal Embedding LLM
    case "MXBAI":
        dir = "../../brain_DB_MXBAI"
        emb_func = OllamaEmbeddings(model = 'mxbai-embed-large', temperature = 0) #recommended by Ollama for embedding textual data

col_name = 'brain_gen_2_panels'
vectorstore = Chroma(persist_directory = dir, 
                     embedding_function = emb_func,
                     collection_name = col_name)
retriever = vectorstore.as_retriever()

#Models that successfully run the full graph:
#################################
model_gemma = ChatOllama(model = "gemma3:4b", temperature = 0)
model_llama = ChatOllama(model = 'llama3.1:8b', temperature = 0)
model_granite = ChatOllama(model = 'granite3.2:8b', temperature = 0)


def get_current_model (sel_model):
    match sel_model.model:
        case ('gemma3:4b'):
            return 'gemma3-4b'
        case ('llama3.1:8b'):
            return 'llama3.1-8b'
        case ('granite3.2:8b'):
            return ('granite3.2-8b')
        
sel_model = model_llama
model_name = get_current_model(sel_model)
print(model_name)


#Have a way to separate if data is coming from image source or text source
def decode_base64_image(base64_string):
    """Decode a Base64 string into a PIL.Image.Image object."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False

def reshape_results(docs):
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            images.append(
                decode_base64_image(doc)
            )  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
structured_llm_grader = sel_model.with_structured_output(GradeDocuments)

# Prompt
system1 = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

ret_system_prompt = system1
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ret_system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

#import all markdown files that are golden examples of test cases
path_tc_folder = '../../inputs/example_tcs'
tc_filenames = os.listdir(path_tc_folder)

all_tc_paths = []
for i in range(len(tc_filenames)):
    if (not tc_filenames[i].startswith(".")):
        all_tc_paths.append(f'{path_tc_folder}/{tc_filenames[i]}')

all_tc_contents = []
all_tc_metadata = []

for path in all_tc_paths:
    print(path)
    content = open(path).read().strip()
    all_tc_contents.append(content)
    all_tc_metadata.append({"source": path})


template_ex_TC_from_file = """
<INSTRUCTIONS>
You are a test engineer that generates test cases. 
For each test case, provide a brief description of the test case and the test steps.
Every test step must have an expected result of what should be displayed on the screen.
Do not use personal pronouns in the test case. 
The results of a search of a database of screenshots of the application have been provided to give you more context. 
Use the information in the results to help you generate test cases for the specified features accurately.
Not all information in the results will be useful. 
Write as many test cases as needed to completely verify the test case objectives. 
However, if you find any information that's useful for generating a test case for the feature specified by the user, draw from it in your answer. 
<EXAMPLE> {example1} </EXAMPLE>
<EXAMPLE> {example2} </EXAMPLE>
</INSTRUCTIONS>

<FEATURE> {feature} </FEATURE>

<RESULTS> {context} </RESULTS>

ANSWER:
"""


template = template_ex_TC_from_file

def gen_TC_w_ex_from_files(feature, context, model = sel_model):
    global template
    template = template_ex_TC_from_file
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    example1 = all_tc_contents[0]
    example2 = all_tc_contents[1]
    example3 = all_tc_contents[2]
    
    return chain.invoke({"example1": example1, "example2": example2, "feature": feature, "context": context})

numb_ret = 10
output_path = './outputs/'
def output_TC(feature, retrieved_info, answer, iter):
    global numb_ret
    global ret_system_prompt
    global logtime
    model_name = get_current_model(sel_model)
    # logtime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    file = open(f'{output_path}{str(logtime)}/AIAgent_{model_name}_{iter}.md', 'w')
    file.write("Date: " + logtime +"\n\n")
    file.write("Retriever settings: retriever" + chosen_emb_model + ", k=" + str(numb_ret) +"\n\n")
    file.write("Model used to infer: " + model_name +"\n\n")
    file.write("Feature: " + feature + "\n\n")
    file.write("Prompt to score the relevance of retrieved documents: " + ret_system_prompt + "\n\n")
    file.write("Prompt to Generate TCs: " + template + "\n\n")
    file.write("================================================\n\n")
    file.write('# FILTERED RETRIEVED INFORMATION FROM VECTOR STORE \n\n')
    for image in retrieved_info["images"]:
        file.write(image + '\n\n')
    for text in retrieved_info["texts"]:
        file.write(text+'\n\n')
    file.write("================================================\n\n")
    file.write("# GENERATED TEST CASES: \n\n")
    file.write(answer)
    file.close()


# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | sel_model | StrOutputParser()

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: history of all messages passed to the LLM
        question: question
        generation: LLM generation
        documents: list of documents
    """
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    new_question: str
    generation: str
    documents: List[str]
    new_documents: List[str]
    iter: int

memory = MemorySaver()

def print_raw_retrieved_doc(prompt, results, iter):
    global numb_ret
    global chosen_emb_model
    global logtime
    model_name = get_current_model(sel_model)
    # logtime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    file = open(f'{output_path}{str(logtime)}/AIAgent_{model_name}_rawRetrievedDocs_{iter}.md', 'w')
    file.write("Date: " + logtime +"\n\n")
    file.write("================================================\n\n")
    file.write('# RAW RETRIEVED INFORMATION FROM VECTOR STORE \n\n')
    file.write("Retriever settings: retriever" + chosen_emb_model + ", k=" + str(numb_ret) +"\n\n")
    file.write("Model used to infer: " + model_name +"\n\n")
    file.write("Prompt: " + prompt +"\n\n")
    file.write("================================================\n\n")
    for result in results:
        try: #if a doc
            file.write(result.metadata["source"] + " " + str(result.metadata['start_index']) + "\n\n")
            file.write(result.page_content + "\n\n")
        except: #if an image
            file.write(result.metadata["source"] + "\n\n")
    file.close()


def retrieve(state):
    global numb_ret
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    
    iter = state["iter"]

    if iter == 0 or iter == None:
        iter = 1
        question = state["question"]
        documents = retriever.invoke(question, k=numb_ret)
        print_raw_retrieved_doc(question, documents, str(iter))
        return {"question": question, "documents": documents, "iter": iter}
    else:
        iter = state["iter"] + 1
        iter = iter + 1
        new_question = state["new_question"]
        new_documents = retriever.invoke(new_question, k=numb_ret) 
        print_raw_retrieved_doc(new_question, new_documents, str(iter))
        return {"new_question": new_question, "new_documents": new_documents, "iter": iter}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question = state["question"]
    documents = state["documents"]
    iter = state["iter"]

    if iter > 1:
        new_question = state["new_question"]
        question = question + " and " + new_question
        new_documents = state["new_documents"]

        #Append only unique new documents to the documents List; Exclude any duplicates
        for i in range(len(new_documents)):
            if new_documents[i] not in documents:
                documents.append(new_documents[i])

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {"question": question, "documents": filtered_docs, "new_documents": None}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        New key added to state, generation, that contains LLM generation
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    processed_docs = reshape_results(documents)

    # RAG generation
    template = template_ex_TC_from_file
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | sel_model
    example1 = all_tc_contents[0]
    example2 = all_tc_contents[1]
    example4 = all_tc_contents[3]
    example5 = all_tc_contents[4]
    example6 = all_tc_contents[5]
    example7 = all_tc_contents[6]
    example8 = all_tc_contents[7]
    
    answer = chain.invoke({"example1": example1, "example2": example2, "example3": example5,
                           "feature": question, "context": processed_docs})

    iter = state["iter"]

    output_TC(question, processed_docs, answer.pretty_repr(), str(iter))
    return {"documents": documents, "question": question, "generation": answer}

def hallucination_checker(state):
    """ 
    Given the Generated Test Case, 
    Check that the content is grounded in facts.

    Args: 
        state (dict): The current graph state
    
    Returns:
        state (dict): Updates new_question key with key terms of specs that are missing from test case.
    """
    print("---EVALUATING TEST CASE CONTENT FOR HALLUCINATIONS---")
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score
    
    if grade == "yes":
        return {"generation": "hallucinations"}
    else:
        return {"generation": "no_hallucinations"}
    

def human_feedback(state):
    """
    Given the Generated Test Case, 
    Ask the User if there is any missing context that shoud be retrieved from the vectorstore

    Args: 
        state (dict): The current graph state
    
    Returns:
        state (dict): Updates new_question key with key terms of specs that are missing from test case.
    """

    print("---EVALUATING TEST CASE CONTENT FOR COMPLETENESS---")
    

    missing_info = input("Enter any information that is missing as input to the test case that was generated. Or type 'None':")
    print("Human said the missing feedback is: ", missing_info)
    return{"new_question": missing_info}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # If No documents are related to the feature, then we will rephrase the feature to generate test cases on. 
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    
def decide_if_hallucinated(state):
    generated_tc = state["generation"]
    if generated_tc == "no_hallucinations":
        print("---DECISION: NO HALLUCINATIONS---")
        return "human_feedback"
    else:
        print("---DECISION: HALLUCINATIONS ARE PRESENT, REGENERATE A RESPONSE")
        return "generate"

def decide_to_approve(state):
    new_query = state["new_question"]
    
    if new_query != "None":
        print("---RETRIEVE MISSING SPECS FROM DATABASE---")
        return "retrieve"
    else:
        print("--END--")
        return "end"

workflow = StateGraph(State)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("human_feedback", human_feedback) #review the TC generated and ensure there was no missing inputs

# Define the edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    }
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", "human_feedback")
workflow.add_conditional_edges(
    "human_feedback", 
    decide_to_approve,
    {
        "retrieve": "retrieve",
        "end": END,
    }
)

# Compile/Build the graph
graph = workflow.compile(interrupt_before=["human_feedback"])

logtime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
folder_path = "outputs/" + logtime
if not os.path.exists(folder_path):
    # If it doesn't exist, create it
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")
