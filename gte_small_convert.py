import coremltools as ct
import numpy as np
import torch
import transformers

from gte_small import BertModel

BATCH_SIZE = 32
SEQ_LENGTH = 128

if __name__ == "__main__":
    model_name = "Supabase/gte-small"
    baseline_model = transformers.AutoModel.from_pretrained(
        model_name,
        return_dict=False,
        torchscript=True,
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    print(baseline_model.config)
    optimized_model = BertModel(baseline_model.config).eval()
    optimized_model.load_state_dict(baseline_model.state_dict())
    tokenized = tokenizer(
        BATCH_SIZE * ["Sample input text to trace the model"],
        return_tensors="pt",
        max_length=SEQ_LENGTH,  # token sequence length
        padding="max_length",
    )
    print(f"INPUT SHAPES: {tokenized['input_ids'].shape}  ")
    traced_optimized_model = torch.jit.trace(
        optimized_model,
        (
            tokenized["input_ids"],
            tokenized["token_type_ids"],
            tokenized["attention_mask"],
        ),
    )
    ane_mlpackage_obj = ct.convert(
        traced_optimized_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                f"input_{name}",
                shape=tensor.shape,
                dtype=np.int32,
            )
            for name, tensor in tokenized.items()
        ],
        compute_units=ct.ComputeUnit.ALL,
    )
    out_path = f"gte_small_seqLen{SEQ_LENGTH}_batchSize{BATCH_SIZE}.mlpackage"
    ane_mlpackage_obj.save(out_path)
