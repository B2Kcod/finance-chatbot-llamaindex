import numpy as np
import litellm
from trulens.core import TruSession, Feedback
from trulens.providers.litellm import LiteLLM
from trulens.apps.llamaindex import TruLlama
from trulens.dashboard import run_dashboard

def start_dashboard():
    session = TruSession()
    run_dashboard(session)

def setup_trulens_recorder(router_query_engine):

    litellm.set_verbose = False
    provider = LiteLLM(model_engine=f"ollama/gemma2:2b")

    # Select context to be used in feedback. The location of context is app specific.
    context = TruLlama.select_context(router_query_engine)

    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons, name="Groundedness"
        )
        .on(context.collect())  # collect context chunks into a list
        .on_output()
    )

    # Question/answer relevance between overall question and answer.
    f_answer_relevance = Feedback(
        provider.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()

    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(
            provider.context_relevance_with_cot_reasons, name="Context Relevance"
        )
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )

    tru_recorder = TruLlama(
        router_query_engine,
        app_id="finance_chatbot_llamaindex",
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
    )
    return tru_recorder
