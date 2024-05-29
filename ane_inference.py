from time import perf_counter

import coremltools as ct
import numpy as np
import transformers

N = 10
BATCH_SIZE = 32
SEQ_LENGTH = 128
base_model_name = "Supabase/gte-small"
coreml_model = f"gte_small_seqLen{SEQ_LENGTH}_batchSize{BATCH_SIZE}.mlpackage"

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name)
    # Load the coreML model
    s = perf_counter()
    print(f"--- Loading tokenizer + model {base_model_name}")
    model = ct.models.MLModel(coreml_model)
    print(f"Time to load model {perf_counter()-s:.2f}s")
    # Input
    tokenized = tokenizer(
        BATCH_SIZE * ["Sample input text to trace the model"],
        return_tensors="np",
        max_length=128,  # token sequence length
        padding="max_length",
    )

    print("--- Starting prediction")
    s = perf_counter()
    for _ in range(N):
        predictions = model.predict(
            {
                "input_input_ids": tokenized["input_ids"].astype(np.int32),
                "input_token_type_ids": tokenized["token_type_ids"].astype(np.int32),
                "input_attention_mask": tokenized["attention_mask"].astype(np.int32),
            }
        )
    print(
        f"Avg time to run prediction ({BATCH_SIZE}x{SEQ_LENGTH}) {1000*((perf_counter()-s)/N):.3f}ms"
    )
    print(
        f"Avg time per sentence ({SEQ_LENGTH}) {1000*((perf_counter()-s)/(N*BATCH_SIZE)):.3f}ms"
    )
    print(f"Embedding shape : {predictions[list(predictions.keys())[-1]].shape}")
