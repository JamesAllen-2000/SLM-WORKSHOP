"""
=============================================================================
 CENTRAL CONFIGURATION - every script in this workshop imports from here.
=============================================================================
 If you want to change the model, the prompts, or how RAG behaves, this is
 the ONLY file you need to edit. Nothing else hard-codes these values.
=============================================================================
"""
import os

# =============================================================================
# SECTION 1: WHICH MODEL ARE WE USING?
# =============================================================================
# The workshop ships with Gemma-3 1B (instruction-tuned). The commented lines
# are alternatives you can experiment with AFTER the workshop - each one needs
# a fresh run of 00 -> 01 -> 02 -> 03 to rebuild all the artifacts.

ACTIVE_MODEL_ID = "google/gemma-3-1b-it"
# ACTIVE_MODEL_ID = "Qwen/Qwen3-1.7B"
# ACTIVE_MODEL_ID = "Qwen/Qwen2.5-1.5B"
# ACTIVE_MODEL_ID = "Qwen/Qwen2.5-0.5B"

# =============================================================================
# SECTION 2: WHERE THINGS LIVE ON DISK
# =============================================================================
def get_safe_model_name():
    """
    Turns a long HuggingFace repo id into a very short folder name.

    "google/gemma-3-1b-it" -> "it"     (so the model lands in models/it)

    Why so short? Windows has a 260-character path limit (MAX_PATH). Model
    folders contain deeply nested files, so a long project path plus a long
    model name overflows it and the download fails.
    """
    return ACTIVE_MODEL_ID.split("-")[-1]


# The base model downloaded by 00_download_base.py
LOCAL_MODEL_DIR = f"models/{get_safe_model_name()}"

# --- RAG storage ------------------------------------------------------------
# Built once by 01_rag_baseline.py, then read by every RAG demo.
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "workshop_knowledge"

# --- Embedding model (turns text into vectors for RAG search) ---------------
# This is a SECOND, much smaller model (~90 MB) separate from the LLM.
# It is downloaded alongside the base model in 00_download_base.py so the
# workshop stays fully offline - see get_embedder_path() below.
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_DIR = "models/embedder"


def trim_runaway(answer):
    """
    Cut a looping answer off at the point it starts repeating itself.

    WHY THIS EXISTS - and it is worth explaining to participants:

    A 1-billion-parameter model sometimes gets stuck in a loop. It finishes a
    perfectly good [ ACTION PLAN ], then instead of stopping it invents
    "[ ACTION PLAN DETAILS ]" and writes the same five steps again, and again,
    until it runs out of tokens. Bigger models do this far less.

    We keep the first, good copy of each section and drop the repeats. The
    answer stays complete - we are cutting duplicated tail, not information.

    Returns (trimmed_answer, lines_removed) so the caller can be honest on
    screen about the fact that trimming happened.
    """
    import difflib
    import re

    expected = ("[ SUMMARY ]", "[ POTENTIAL CAUSES ]", "[ ACTION PLAN ]")

    def normalise(text):
        """Strip list markers and filler so near-identical steps compare equal."""
        t = text.strip().lower()
        t = re.sub(r"^[-*•]\s*", "", t)        # bullet
        t = re.sub(r"^\d+[.)]\s*", "", t)           # "3. " / "3) "
        t = re.sub(r"^(the|a|an)\s+", "", t)        # leading article
        # "check" / "verify" / "inspect" are interchangeable in these answers
        t = re.sub(r"^(check|verify|inspect|review|confirm)\s+", "", t)
        return re.sub(r"[^a-z0-9 ]", "", t)

    lines = answer.split("\n")
    kept = []
    seen_headers = set()
    seen_bodies = []          # normalised content lines we have already kept

    for line in lines:
        stripped = line.strip()

        # --- Section headers look like "[ SOMETHING ]" ----------------------
        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped.upper()

            # A header we have already produced = the model has looped.
            if header in seen_headers:
                break

            # A header that is not one of the three we asked for, appearing
            # after we already wrote the action plan, is invented padding
            # ("[ ACTION PLAN DETAILS ]"). Stop there.
            if header not in expected and "[ ACTION PLAN ]" in seen_headers:
                break

            seen_headers.add(header)
            kept.append(line)
            continue

        # --- Content lines: drop ones we have effectively already said ------
        # The model often restates a cause or a step with one word changed
        # ("Check pump alignment..." then "Verify pump alignment...").
        # Short lines are left alone; they are rarely the problem and are
        # sometimes legitimately similar.
        norm = normalise(stripped)
        if len(norm) > 25:
            if any(difflib.SequenceMatcher(None, norm, prev).ratio() > 0.90
                   for prev in seen_bodies):
                continue
            seen_bodies.append(norm)

        kept.append(line)

    # Renumber any list that lost entries, so we do not show "1. 2. 5. 6."
    out, n = [], 0
    for line in kept:
        m = re.match(r"^(\s*)\d+([.)]\s+)(.*)$", line)
        if m:
            n += 1
            out.append(f"{m.group(1)}{n}{m.group(2)}{m.group(3)}")
        else:
            if line.strip().startswith("["):
                n = 0        # new section restarts numbering
            out.append(line)

    removed = len(lines) - len(out)
    return "\n".join(out).rstrip(), removed


def get_embedder_path():
    """
    Prefer the local offline copy; fall back to the HuggingFace Hub id.

    Without this, every RAG script would reach out to the internet on startup
    to fetch the embedding model - which defeats the whole point of an
    "offline edge device" demo (and melts the wifi when 40 people run it at
    the same time).
    """
    return EMBEDDING_MODEL_DIR if os.path.isdir(EMBEDDING_MODEL_DIR) else EMBEDDING_MODEL_ID


# =============================================================================
# SECTION 3: HOW DOCUMENTS ARE SPLIT FOR RAG  (used by 01_rag_baseline.py)
# =============================================================================
# Documents are too long to feed to a 1B model whole, so we cut them into
# overlapping pieces ("chunks"). The overlap stops us from slicing a sentence
# in half and losing its meaning.
#
# NOTE: these are CHARACTER counts, not tokens. 1024 characters is roughly
# 250-300 tokens for English technical text.
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 150

# =============================================================================
# SECTION 4: HOW THE MODEL GENERATES TEXT
# =============================================================================
TEMPERATURE = 0.3        # 0.0 = always pick the likeliest word (repeatable)
                         # 1.0 = creative and unpredictable.
                         # 0.3 keeps technical answers consistent.
TOP_P = 0.9              # Only sample from the most likely 90% of words.
MAX_TOKENS = 450         # Longest answer we allow. Every demo uses this same
                         # value so their speeds are directly comparable.
                         #
                         # DO NOT RAISE THIS. Measured on the standard pump
                         # question:
                         #   450 -> finishes on its own at ~262 tokens, ending
                         #          on a complete sentence with real readings.
                         #   800 -> runs to the limit and degenerates, padding
                         #          the action plan out to 38 repeated steps.
                         #
                         # The cap is not cutting answers short; it is stopping
                         # a small model from rambling. If an answer really is
                         # truncated, reduce TOP_K instead so less context
                         # competes for the window.
REPEAT_PENALTY = 1.05    # Gently discourages the model repeating itself.
                         #
                         # MEASURED, on the standard pump question:
                         #   1.05 -> 1349 chars, quotes 7 real readings
                         #   1.15 ->  802 chars, quotes the same 7 readings
                         #
                         # Both stay grounded; 1.15 just strips the explanation
                         # around the numbers. We keep 1.05 because a fuller
                         # answer is better teaching material, and the workshop
                         # is not in a hurry.
                         #
                         # TRADE-OFF: at 1.05 this 1B model will occasionally
                         # repeat an action-plan step, or loop until it hits
                         # MAX_TOKENS. That is a real property of small models
                         # and worth showing. If it happens mid-demo, just ask
                         # again - or raise this to 1.15 for a terser, tighter
                         # answer.
N_CTX = 4096             # Context window: prompt + retrieved docs + answer
                         # must all fit inside this. Raise to 8192 if answers
                         # get cut off mid-sentence.
TOP_K = 3                # How many document chunks RAG retrieves per question.

# --- Demo visibility --------------------------------------------------------
# Show the actual document text RAG pulled in, before the model answers.
# This is the single most useful thing to have on screen during the workshop:
# participants can see the real readings and history in the retrieved chunks,
# and then watch which of them the model actually uses in its answer.
# Set False for a cleaner screen once people have seen it a few times.
SHOW_RETRIEVED_CONTEXT = True
CONTEXT_PREVIEW_CHARS = 420   # Per chunk. Raise to see more of each document.

DETERMINISTIC_MODE = False   # Set True to force temperature 0 - useful when
                             # you want the same answer every single run.

# =============================================================================
# SECTION 5: RAG CONFIDENCE THRESHOLDS
# =============================================================================
# ChromaDB returns a "distance" for each retrieved chunk: SMALLER = more
# similar to the question. Because our embedding model produces unit-length
# vectors, distance relates to similarity as:  distance = 2 - 2 x cosine
#
#   distance 0.35  ->  ~83% similar   (near-perfect match)
#   distance 0.65  ->  ~68% similar   (clearly relevant)
#   distance 1.00  ->  ~50% similar   (weak - last useful cutoff)
#
# We pick a different system prompt depending on how good the match was.
# If you swap EMBEDDING_MODEL_ID for a model that does NOT normalise its
# vectors, these numbers stop being meaningful and must be re-tuned.
EXACT_MATCH_THRESHOLD = 0.35     # Trust the documents almost completely.
STRONG_MATCH_THRESHOLD = 0.65    # Use documents, but reason around them.
NO_MATCH_THRESHOLD = 1.0         # Beyond this, ignore RAG entirely and fall
                                 # back to the model's own knowledge.

# =============================================================================
# SECTION 6: SYSTEM PROMPTS
# =============================================================================
# A "system prompt" is a standing instruction the model reads before every
# question. We use three, chosen automatically by how well RAG matched:
#
#   BASE_SYSTEM_PROMPT       - no documents available; answer from training.
#   RAG_SYSTEM_PROMPT        - documents found; ground the answer in them.
#   EXACT_RAG_SYSTEM_PROMPT  - excellent match; extract almost verbatim.
#
# WORKSHOP TALKING POINT: these prompts are long (roughly 560, 720 and 425
# tokens). On an edge CPU the model must read every one of those tokens
# before writing a single word of the answer - about 3.5 seconds of pure
# prompt-reading at ~157 tokens/sec. That is the cost fine-tuning is meant
# to buy back, by baking the behaviour into the weights instead.

BASE_SYSTEM_PROMPT = """
You are an offline oil-field field-service decision-support assistant running
locally on an edge device.

Your job is to give technicians clear, technically useful, concise answers.

GENERAL RULES
- Respond professionally and conversationally.
- Give the answer immediately. Do not add greetings.
- Do not use markdown bolding or asterisks.
- Do not invent equipment readings, limits, alarm meanings, specifications,
  procedures, or site-specific values.
- Do not repeat the user's question.
- Prefer practical information over generic explanations.
- Keep answers focused and avoid unnecessary background information.

--------------------------------------------------
RESPONSE MODE SELECTION
--------------------------------------------------

Determine the type of question before answering.

MODE 1 — DEFINITION / GENERAL KNOWLEDGE

Use this mode for questions such as:
- "What is pressure?"
- "What is a centrifugal pump?"
- "What is cavitation?"
- "What does a bearing do?"

Response requirements:
- Answer directly in 2 to 4 sentences.
- Explain what it is.
- Briefly explain its relevance to oil-field operations when useful.
- Do NOT include troubleshooting steps.
- Do NOT include sections such as SUMMARY, CAUSES, or ACTION PLAN.

MODE 2 — TROUBLESHOOTING / EQUIPMENT PROBLEM

Use this mode when the user describes:
- an alarm
- equipment failure
- abnormal vibration
- overheating
- abnormal pressure
- abnormal flow
- repeated shutdown
- electrical fault
- mechanical fault
- maintenance problem

Use exactly this structure:

[ SUMMARY ]
Give a concise 2-3 sentence assessment of the reported problem.
Mention the equipment, important symptom, and what type of fault it may indicate.
Do not claim a root cause unless it is reasonably supported.

[ POTENTIAL CAUSES ]
List 3-5 likely causes, ordered from most important to less likely.

For each cause:
- name the cause
- briefly explain why it could produce the reported symptom

[ ACTION PLAN ]
Give 4-6 numbered troubleshooting actions.

Order the actions by:
1. safety
2. easiest/highest-value checks
3. electrical checks
4. mechanical checks
5. deeper inspection

Each action must tell the technician what to inspect or verify.

If an exact value or limit is unknown, say to compare it with the
equipment/site specification rather than inventing a number.

If the available information is insufficient to identify the root cause,
state that clearly while still providing useful initial checks.

Do not output anything before [ SUMMARY ] or after [ ACTION PLAN ].
"""


RAG_SYSTEM_PROMPT = """
You are an offline oil-field field-service decision-support assistant.

The user is working in the field and may have no internet connection.

You will receive RETRIEVED CONTEXT containing information from local:
- equipment manuals
- troubleshooting guides
- maintenance records
- historical incidents
- service tickets
- operating procedures

Your answer must be grounded in that retrieved context.

--------------------------------------------------
GROUNDING RULES
--------------------------------------------------

1. Use the retrieved context as the primary source of truth.
2. Extract all useful information relevant to the user's problem.
3. Do not invent:
   - measurements
   - alarm meanings
   - operating limits
   - equipment specifications
   - component names
   - previous incidents
   - maintenance actions
4. Use exact numerical values only when they are present in the context.
5. Do not assume that a historical cause is automatically the current cause.
6. Clearly distinguish between:
   - confirmed information
   - likely causes
   - recommended checks
7. Ignore retrieved information that is unrelated to the user's problem.
8. Do not mention RAG, embeddings, vector databases, similarity scores,
   retrieved documents, or the AI system.
9. Do not use markdown bolding or asterisks.
10. Do not begin with phrases such as:
    - "Based on the context"
    - "According to the retrieved documents"
    - "Here's a breakdown"

--------------------------------------------------
REQUIRED OUTPUT FORMAT
--------------------------------------------------

Always use exactly these three sections for troubleshooting questions:

[ SUMMARY ]

Write 2-4 sentences.

Include:
- the equipment or component involved
- the reported symptom/alarm
- what category of problem the evidence suggests
- the most important immediate concern

Do not declare a definite root cause unless the context confirms it.

[ POTENTIAL CAUSES ]

Provide 3-5 causes supported by the retrieved information.

Order them from highest troubleshooting priority to lowest.

Format:

- Cause: short explanation of why it matches the symptoms.
- Cause: short explanation of why it matches the symptoms.

Whenever possible, connect the cause to evidence such as:
- abnormal current
- temperature
- pressure
- flow
- vibration
- voltage
- maintenance history
- previous failures

Do not include causes unsupported by the context.

[ ACTION PLAN ]

Provide 4-6 numbered actions.

The first actions must be the safest and fastest checks that help isolate
the problem.

Each step should contain:
- what to check
- what observation to look for
- what the result would indicate when the context supports that conclusion

Use available equipment-specific values from the context.

Example style:

1. Review the overload/thermal trip history and confirm whether the trip
   coincides with elevated motor temperature or current.
2. Measure the three-phase voltage and compare the readings for imbalance.
3. Check pump and motor alignment for evidence of increased mechanical load.

Do not simply write generic instructions such as:
"Inspect the equipment" or "Check the pump."

If the context does not provide enough evidence for a definitive diagnosis,
end the final action with the next logical inspection instead of inventing
a diagnosis.

Do not output anything before [ SUMMARY ].
Do not output anything after the final ACTION PLAN step.
"""


EXACT_RAG_SYSTEM_PROMPT = """
You are an offline oil-field field-service decision-support assistant.

The retrieved context contains highly relevant information for the user's
question.

Your task is to extract and organize that information accurately.

--------------------------------------------------
STRICT GROUNDING RULES
--------------------------------------------------

- Use only information explicitly supported by the retrieved context.
- Do not add technical facts from general knowledge.
- Do not invent measurements, limits, procedures, component names, or causes.
- Preserve equipment identifiers exactly as provided.
- Preserve important numerical values and units accurately.
- Combine duplicate information from multiple context chunks.
- Prioritize information directly related to the user's reported symptom.
- Do not mention retrieved context, documents, RAG, similarity scores,
  embeddings, or vector databases.
- Do not use markdown bolding or asterisks.

--------------------------------------------------
REQUIRED OUTPUT
--------------------------------------------------

[ SUMMARY ]

Write 2-4 sentences describing:
- the reported equipment problem
- the most relevant evidence
- what the available information indicates

Only state a root cause as confirmed when the context explicitly confirms it.

[ POTENTIAL CAUSES ]

List all relevant causes contained in the context, up to 5.

Format each item as:

- Cause: evidence or reason from the context.

If the context does not identify a cause, write:

- The retrieved information does not identify a confirmed cause.

[ ACTION PLAN ]

Provide 4-6 numbered actions derived from the context.

Prioritize:
1. safety-related checks
2. alarm/trip verification
3. operating measurements
4. electrical inspection
5. mechanical inspection
6. maintenance/escalation actions

Include exact limits, measurements, tolerances, or equipment identifiers
when they appear in the context.

If a required troubleshooting action is not present in the context,
do not invent one.

Do not output anything before [ SUMMARY ].
Do not output anything after the final ACTION PLAN step.
"""

