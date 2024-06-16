-- DROP TRIGGERS
DROP TRIGGER IF EXISTS update_presets_updated_at ON presets;
DROP TRIGGER IF EXISTS update_preset_workspaces_updated_at ON preset_workspaces;

-- DROP POLICIES
DROP POLICY IF EXISTS "Allow full access to own presets" ON presets;
DROP POLICY IF EXISTS "Allow view access to non-private presets" ON presets;
DROP POLICY IF EXISTS "Allow full access to own preset_workspaces" ON preset_workspaces;

-- DROP INDEXES
DROP INDEX IF EXISTS presets_user_id_idx;
DROP INDEX IF EXISTS preset_workspaces_user_id_idx;
DROP INDEX IF EXISTS preset_workspaces_preset_id_idx;
DROP INDEX IF EXISTS preset_workspaces_workspace_id_idx;

-- DROP DEPENDENT OBJECTS
ALTER TABLE preset_workspaces DROP CONSTRAINT IF EXISTS preset_workspaces_preset_id_fkey;
ALTER TABLE preset_workspaces DROP CONSTRAINT IF EXISTS preset_workspaces_workspace_id_fkey;
ALTER TABLE preset_workspaces DROP CONSTRAINT IF EXISTS preset_workspaces_user_id_fkey;

ALTER TABLE presets DROP CONSTRAINT IF EXISTS presets_folder_id_fkey;
ALTER TABLE presets DROP CONSTRAINT IF EXISTS presets_user_id_fkey;

-- DROP TABLES
DROP TABLE IF EXISTS preset_workspaces;
DROP TABLE IF EXISTS presets;