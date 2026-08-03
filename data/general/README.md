# General-capability data notice

Generated files under `data/general/raw` and `data/general/processed` are excluded from Git.

The general-language stage uses [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext) at revision `b08601e04326c79dfdd32d625aee71d232d685c3`, configuration `wikitext-103-raw-v1`. WikiText is distributed under CC BY-SA 3.0 and GFDL terms.

The instruction stage uses [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) at revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`. Dolly is distributed under CC BY-SA 3.0.

`cleo-1 prepare-general` verifies pinned byte sizes and SHA-256 hashes, encodes WikiText with the existing lossless Cleo 1 tokenizer, and creates deterministic category-stratified Dolly train, validation, and test splits. The generated manifest records source provenance and processed checksums.
