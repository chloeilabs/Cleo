export interface ModelProfile {
  identity: {
    company_name: string
    model_name: string
    model_id: string
    release: string
  }
  runtime: {
    device: string
    checkpoint: string
    saved_at_utc: string
  }
  metrics: {
    parameter_count: number
    training_step: number
    initial_validation_loss: number
    best_validation_loss: number
    best_validation_perplexity: number
    loss_reduction_percent: number
  }
  architecture: {
    block_size: number
    vocab_size: number
    n_layer: number
    n_head: number
    n_embd: number
    ffn_size: number
    dropout: number
  }
  training: {
    duration: string
    elapsed_seconds: number
    tokens_seen: number
  }
  adaptation: {
    identity_tuned: boolean
    completed_steps: number
    held_out_exact_match: number
    story_loss_ratio: number
    deterministic_api_identity: boolean
  }
  dataset: {
    name: string
    revision: string
    license: string
    train_stories: number
    validation_stories: number
    train_tokens: number
    validation_tokens: number
  }
  benchmark: {
    device: string
    cached_tokens_per_second: number
    uncached_tokens_per_second: number
    cache_speedup: number
    new_tokens: number
    outputs_equal: boolean
  }
  validation_curve: Array<{ step: number; loss: number }>
  samples: Array<{ prompt: string; seed: number; text: string }>
  prompt_starters: Array<{ label: string; prompt: string }>
}

export interface GenerationRequest {
  prompt: string
  max_new_tokens: number
  temperature: number
  top_k: number
  seed: number
}

export interface GenerationEvent {
  type: "generation"
  text: string
  status: string
}
