"""
eval_dataset.py - Golden Q&A set for evaluating the insurance claim support agent.

Each entry has:
- question: the user query
- expected_source: the data/ filename the retriever SHOULD surface this
  chunk from (used for retrieval hit-rate / MRR)
- reference_answer: the key facts a correct answer should contain (used by
  the LLM-as-judge for answer quality, NOT shown to the agent)
"""

EVAL_QUESTIONS = [
    # --- Auto policy ---
    {
        "question": "What's my deductible for collision damage?",
        "expected_source": "auto_insurance_policy.md",
        "reference_answer": "The standard collision deductible is $500. Options of $250 or $1,000 are also available.",
    },
    {
        "question": "What are my bodily injury liability limits on my auto policy?",
        "expected_source": "auto_insurance_policy.md",
        "reference_answer": "$100,000 per person and $300,000 per accident.",
    },
    {
        "question": "Is my car covered if I use it for food delivery?",
        "expected_source": "auto_insurance_policy.md",
        "reference_answer": "No. Ridesharing or commercial delivery use is excluded unless a commercial rider is added.",
    },
    {
        "question": "How much does my policy pay for a rental car while my vehicle is being repaired?",
        "expected_source": "auto_insurance_policy.md",
        "reference_answer": "$30 per day, up to a maximum of 30 days.",
    },
    {
        "question": "What's covered under roadside assistance and what's the limit?",
        "expected_source": "auto_insurance_policy.md",
        "reference_answer": "Towing, flat tire change, jump-start, and lockout service, up to $100 per occurrence.",
    },

    # --- Home policy ---
    {
        "question": "What's the dwelling coverage limit on my home policy?",
        "expected_source": "home_insurance_policy.md",
        "reference_answer": "$300,000.",
    },
    {
        "question": "Is flood damage covered by my homeowners policy?",
        "expected_source": "home_insurance_policy.md",
        "reference_answer": "No, flood damage is excluded and requires a separate flood policy (e.g. through the NFIP).",
    },
    {
        "question": "What's my deductible for wind or hail damage on my home?",
        "expected_source": "home_insurance_policy.md",
        "reference_answer": "2% of the dwelling coverage limit, which works out to $6,000 in designated high-risk regions.",
    },
    {
        "question": "How much personal property coverage do I have, and is there a limit on jewelry?",
        "expected_source": "home_insurance_policy.md",
        "reference_answer": "$150,000 total personal property coverage, with a $1,500 sub-limit on jewelry and watches.",
    },
    {
        "question": "What does loss of use coverage pay for and what's the limit?",
        "expected_source": "home_insurance_policy.md",
        "reference_answer": "Additional living expenses like temporary housing and meals if the home becomes uninhabitable, up to $60,000.",
    },

    # --- Health plan ---
    {
        "question": "What's my individual deductible on the health plan?",
        "expected_source": "health_insurance_policy.md",
        "reference_answer": "$1,500 per plan year.",
    },
    {
        "question": "What's the copay for a specialist visit?",
        "expected_source": "health_insurance_policy.md",
        "reference_answer": "$50.",
    },
    {
        "question": "What's the family out-of-pocket maximum on my health plan?",
        "expected_source": "health_insurance_policy.md",
        "reference_answer": "$12,000 per plan year.",
    },
    {
        "question": "Do I need prior authorization for an MRI?",
        "expected_source": "health_insurance_policy.md",
        "reference_answer": "Yes, outpatient MRI and CT scans require prior authorization.",
    },
    {
        "question": "What's my coinsurance if I see an out-of-network provider?",
        "expected_source": "health_insurance_policy.md",
        "reference_answer": "The plan pays 60% and the member pays 40% for out-of-network care.",
    },

    # --- Claims FAQ ---
    {
        "question": "Can I choose my own repair shop for an auto claim?",
        "expected_source": "claims_faq.md",
        "reference_answer": "Yes, any licensed repair shop can be used; Harborlight's preferred shop network is optional.",
    },
    {
        "question": "What is subrogation?",
        "expected_source": "claims_faq.md",
        "reference_answer": "The process where Harborlight seeks reimbursement from an at-fault third party after paying a claim, which can result in the deductible being refunded.",
    },
    {
        "question": "Why might my health insurance claim get denied?",
        "expected_source": "claims_faq.md",
        "reference_answer": "Common reasons include missing prior authorization, using an out-of-network provider, an excluded service, or missing information.",
    },

    # --- Claims process guide ---
    {
        "question": "How long does a standard auto collision claim typically take to process?",
        "expected_source": "claims_process_guide.md",
        "reference_answer": "About 7 to 14 business days.",
    },
    {
        "question": "What's the first thing that happens after I report a claim?",
        "expected_source": "claims_process_guide.md",
        "reference_answer": "You receive a claim number in the format CLM-YYYY-XXXXXX along with a confirmation email.",
    },
]
