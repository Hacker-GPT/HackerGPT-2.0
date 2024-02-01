CREATE TABLE IF NOT EXISTS subscriptions (
    -- ID
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    subscription_id TEXT NOT NULL CHECK (char_length(subscription_id) <= 1000),

    -- RELATIONSHIPS
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- METADATA
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ,

    -- REQUIRED
    status TEXT NOT NULL CHECK (char_length(status) <= 1000),
    ended_at TIMESTAMPTZ NULL,

    --- UNIQUE subscription_id
    UNIQUE (subscription_id)
);

