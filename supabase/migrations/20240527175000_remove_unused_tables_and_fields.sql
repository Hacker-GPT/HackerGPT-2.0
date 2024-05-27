DROP TABLE IF EXISTS presets;
DROP TABLE IF EXISTS preset_workspaces;
DROP TABLE IF EXISTS prompts;
DROP TABLE IF EXISTS prompts_workspaces;
DROP TABLE IF EXISTS tool_workspaces;
DROP TABLE IF EXISTS collections;
DROP TABLE IF EXISTS collection_workspaces;
DROP TABLE IF EXISTS collection_files;
DROP TABLE IF EXISTS assistant_workspaces;
DROP TABLE IF EXISTS assistant_tools;
DROP TABLE IF EXISTS assistant_files;
DROP TABLE IF EXISTS assistant_collections;

ALTER TABLE profiles
DROP COLUMN IF EXISTS anthropic_api_key,
DROP COLUMN IF EXISTS azure_openai_api_key,
DROP COLUMN IF EXISTS use_azure_openai,
DROP COLUMN IF EXISTS azure_openai_35_turbo_id,
DROP COLUMN IF EXISTS azure_openai_45_turbo_id,
DROP COLUMN IF EXISTS azure_openai_45_vision_id,
DROP COLUMN IF EXISTS azure_openai_endpoint,
DROP COLUMN IF EXISTS google_gemini_api_key,
DROP COLUMN IF EXISTS mistral_api_key,
DROP COLUMN IF EXISTS openai_api_key,
DROP COLUMN IF EXISTS openai_organization_id,
DROP COLUMN IF EXISTS perplexity_api_key,
DROP COLUMN IF EXISTS openrouter_api_key,
DROP COLUMN IF EXISTS azure_openai_embeddings_id,