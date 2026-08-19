# NovaBank AI Support



An **AI-powered banking customer support and ticket intelligence system** that combines semantic intent classification, FAISS-based retrieval, Retrieval-Augmented Generation (RAG), conversation memory, and automated ticket intelligence.



The system is designed around a fictional bank, **NovaBank**, and demonstrates how modern NLP and RAG techniques can be combined to build an end-to-end customer support workflow.



---



## Features



### AI Customer Support



* Semantic customer-intent classification

* Sentence-transformer embeddings

* FAISS similarity search

* Retrieval-Augmented Generation (RAG)

* Grounded responses using the NovaBank knowledge base

* Multi-turn conversation memory



### Ticket Intelligence



* Automatic intent detection

* Ticket category classification

* Sentiment analysis

* Priority assignment

* AI-generated ticket summaries

* Suggested support actions

* Automatic ticket creation

* Ticket persistence using CSV

* Ticket status management



### Support Dashboard



* Total ticket count

* High/Medium/Low priority breakdown

* Category analytics

* Intent analytics

* Ticket Explorer

* Ticket filtering

* Ticket status updates

* Newly generated tickets appear in the dashboard



---



## System Architecture



```text

Customer Query

&#x20;     |

&#x20;     v

Semantic Intent Classifier

&#x20;     |

&#x20;     v

Query Embedding Model

&#x20;     |

&#x20;     v

FAISS Search

&#x20;     |

&#x20;     v

NovaBank Knowledge Base

&#x20;     |

&#x20;     v

RAG Context

&#x20;     |

&#x20;     v

Groq LLM

&#x20;     |

&#x20;     v

Grounded Support Response

&#x20;     |

&#x20;     +------------------------+

&#x20;     |                        |

&#x20;     v                        v

Conversation Memory    Ticket Intelligence

&#x20;                             |

&#x20;               +-------------+-------------+

&#x20;               |             |             |

&#x20;               v             v             v

&#x20;            Category      Sentiment     Priority

&#x20;               |             |             |

&#x20;               +-------------+-------------+

&#x20;                             |

&#x20;                             v

&#x20;                      Support Ticket

&#x20;                             |

&#x20;                             v

&#x20;                     Analytics Dashboard

```



---



## Tech Stack



### Programming



* Python



### Machine Learning and NLP



* Scikit-learn

* Sentence Transformers

* Transformers

* Logistic Regression

* TF-IDF

* Natural Language Processing



### Retrieval



* FAISS

* Sentence embeddings

* Similarity-based retrieval



### Generative AI



* Groq API

* `openai/gpt-oss-120b`



### Application



* Streamlit



### Data and Persistence



* Pandas

* CSV

* Joblib



### Visualization



* Matplotlib

* Seaborn



---



## Intent Classification



Two approaches were implemented and compared.



### TF-IDF + Logistic Regression



The baseline model achieved:



**Accuracy: 85.45%**



### Sentence Embeddings + Logistic Regression



The semantic approach uses:



```text

all-MiniLM-L6-v2

```



with 384-dimensional sentence embeddings.



The model achieved:



**Accuracy: 90.81%**



This represents an improvement of approximately **5.36 percentage points** over the TF-IDF baseline.



The semantic model was selected as the primary intent classifier because it better captures the meaning of customer queries rather than relying only on word-level features.



---



## RAG Pipeline



The system uses a NovaBank-specific knowledge base containing **18 knowledge documents**.



The documents are converted into sentence embeddings and stored in a FAISS index.



```text

Customer Query

&#x20;     |

&#x20;     v

Sentence Embedding

&#x20;     |

&#x20;     v

FAISS Similarity Search

&#x20;     |

&#x20;     v

Relevant Knowledge

&#x20;     |

&#x20;     v

RAG Prompt

&#x20;     |

&#x20;     v

LLM

&#x20;     |

&#x20;     v

Support Response

```



The RAG approach helps the language model generate responses based on the available NovaBank knowledge instead of relying entirely on its pretrained knowledge.



---



## Conversation Memory



The application maintains conversation history during a customer support session.



Example:



```text

Customer:

I lost my card.



Assistant:

...



Customer:

I have already frozen it.



Assistant:

...



Customer:

Can I get a replacement?

```



The system passes previous conversation turns to the response-generation stage so that follow-up questions can be interpreted in context.



---



## Ticket Intelligence



The ticket intelligence pipeline automatically transforms customer conversations into structured support tickets.



A generated ticket can contain:



```text

Ticket ID

Conversation ID

Intent

Category

Sentiment

Sentiment Confidence

Priority

Summary

Suggested Action

Status

```



### Priority Logic



High-priority intents include issues such as:



* Lost or stolen card

* Unrecognized card payment

* Unrecognized cash withdrawal

* Unable to verify identity



Medium-priority cases include issues such as:



* Pending transfer

* Card not working

* Failed top-up

* Contactless not working



Other supported intents are assigned lower priority according to the implemented rules.



---



## Dashboard



The Streamlit dashboard provides an interface for support teams to:



* Monitor ticket volume

* View priority distribution

* Analyze categories

* Explore individual tickets

* Filter tickets

* Update ticket status

* Generate new tickets from conversations



Ticket changes are persisted to:



```text

data/multi\_tickets.csv

```



Therefore, ticket status changes remain available after restarting the Streamlit application.



---



## Project Structure



```text

AI-Powered Banking Customer Support/

|

â”œâ”€â”€ app.py

â”œâ”€â”€ README.md

â”œâ”€â”€ requirements.txt

â”œâ”€â”€ .gitignore

|

â”œâ”€â”€ 01\_intent\_classification\_and\_rag\_retrieval.ipynb

â”œâ”€â”€ 02\_rag\_response\_generation.ipynb

â”œâ”€â”€ 03\_ticket\_intelligence.ipynb

|

â”œâ”€â”€ data/

|   â””â”€â”€ multi\_tickets.csv

|

â”œâ”€â”€ models/

|   â”œâ”€â”€ embedding\_intent\_classifier.pkl

|   â”œâ”€â”€ intent\_classifier.pkl

|   â”œâ”€â”€ tfidf\_vectorizer.pkl

|   â”œâ”€â”€ novabank\_faiss.index

|   â””â”€â”€ novabank\_knowledge.pkl

|

â””â”€â”€ .streamlit/

&#x20;   â””â”€â”€ secrets.toml

```



> `secrets.toml` contains the API key and is excluded from version control using `.gitignore`.



---



## Installation



Clone the repository and move into the project directory.



Then install the required dependencies:



```bash

pip install -r requirements.txt

```



---



## API Configuration



Create:



```text

.streamlit/secrets.toml

```



and add your Groq API key:



```toml

GROQ\_API\_KEY = "YOUR\_API\_KEY"

```



**Never commit this file to GitHub.**



The repository's `.gitignore` excludes:



```text

.streamlit/secrets.toml

.env

```



---



## Run the Application



From the project directory:



```bash

python -m streamlit run app.py

```



Then open the local Streamlit URL shown in the terminal.



---



## Example Customer Queries



Try queries such as:



```text

I lost my card

```



```text

Someone used my card and I don't recognize the payment

```



```text

My transfer is still pending

```



```text

Why can't I verify my identity?

```



```text

What exchange rate do you use?

```



The system predicts the intent, retrieves relevant banking knowledge, and generates a support response.



---



## Limitations



This project is a portfolio implementation and uses a fictional banking knowledge base.



Some limitations were observed during testing:



* Certain closely related banking intents can be confused by the classifier.

* Generic sentiment models may classify neutral banking questions incorrectly.

* Generated response quality depends on the retrieved knowledge and language model.

* The knowledge base contains a limited number of NovaBank-specific documents.

* Ticket persistence currently uses CSV rather than a production database.

* The system is not connected to real banking accounts or financial systems.



These limitations are intentionally documented rather than hidden.



---



## Future Improvements



Potential future improvements include:



* Fine-tuning a domain-specific sentiment classifier

* Improving intent classification for closely related banking intents

* Expanding the NovaBank knowledge base

* Adding a vector database for larger-scale retrieval

* Replacing CSV persistence with PostgreSQL or another production database

* Adding authentication and role-based access

* Adding real-time ticket notifications

* Adding human-agent escalation

* Deploying the application to a cloud platform

* Adding monitoring and model-performance tracking



---



## Project Objective



The goal of this project is to demonstrate an end-to-end **AI customer support workflow** combining:



**Machine Learning + NLP + Semantic Search + RAG + Generative AI + Ticket Intelligence + Streamlit**



rather than implementing a simple chatbot alone.



---



## Author



**Ashutosh Yadav**



B.Tech CSE (AI/ML)



Interested in:



* Machine Learning

* Data Science

* NLP

* Generative AI

* RAG Systems

* AI Applications




