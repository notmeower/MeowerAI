export interface DiscordConfig {
  token: string;
  clientId: string;
}

export interface LLMConfig {
  name: string;
  provider: string;
  version: string;
  endpoint: string;
  apiKey: string;
  maxTokens: number;
  temperature: number;
  top_p: number;
  do_sample: boolean;
  rejectUnauthorized: boolean;
}

export interface Config {
  discord: DiscordConfig;
  llm: LLMConfig;
}

export interface FetchResponse<T = any> {
  ok: boolean;
  status: number;
  body: T;
}

export interface HealthResponse {
  model_loaded?: boolean;
}

export interface InfoResponse {
  parameters?: number;
  max_seq_length?: number;
  vocab_size?: number;
  device?: string;
}

export interface GeneratePayload {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  do_sample: boolean;
}

export interface GenerateResponse {
  text?: string;
  tokens_generated?: number;
}
