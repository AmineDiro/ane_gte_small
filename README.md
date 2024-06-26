# gte-small inference using ANE
This is a port of the `Supabase/gte-small` model for optimized inference.


## Model conversion
1. Create python environment using your favorite tools and run `pip install -r requirements.txt`
2. Run the following script : 
```bash
# Downloads and converts Supabase/gte-small model to a CoreML optimized model
python gte_small_convert.py 
```
3. Run inference  `python ane_inference.py`

![bfde7be4-bfe1-4f91-a7ca-cc286ca5db57](https://github.com/AmineDiro/ane_gte_small/assets/14312141/067f83ab-cb4f-4f12-a516-fde58f50aef2)


## Reference :
- Original research article: [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
