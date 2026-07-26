PRAGMA foreign_keys = OFF;

CREATE TABLE user_profiles (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL CHECK(length(trim(display_name)) > 0),
    created_at TEXT NOT NULL
);

CREATE TABLE continuities (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL CHECK(length(trim(display_name)) > 0),
    user_profile_id TEXT NOT NULL REFERENCES user_profiles(id),
    created_at TEXT NOT NULL,
    archived_at TEXT,
    UNIQUE(id, user_profile_id)
);

INSERT INTO user_profiles(id, display_name, created_at)
SELECT 'profile:' || id, '利用者', created_at
FROM conversations;

INSERT INTO continuities(id, display_name, user_profile_id, created_at)
SELECT 'continuity:' || id, title, 'profile:' || id, created_at
FROM conversations;

CREATE TABLE conversations_new (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    active_branch_id TEXT,
    character_version_id TEXT NOT NULL REFERENCES character_versions(id),
    model_profile_id TEXT NOT NULL REFERENCES model_profiles(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT,
    auto_translate INTEGER NOT NULL DEFAULT 0
        CHECK (auto_translate IN (0, 1)),
    continuity_id TEXT NOT NULL REFERENCES continuities(id),
    UNIQUE(id, continuity_id)
);

INSERT INTO conversations_new(
    id, title, active_branch_id, character_version_id, model_profile_id,
    created_at, updated_at, archived_at, auto_translate, continuity_id
)
SELECT
    id, title, active_branch_id, character_version_id, model_profile_id,
    created_at, updated_at, archived_at, auto_translate, 'continuity:' || id
FROM conversations;

DROP TABLE conversations;
ALTER TABLE conversations_new RENAME TO conversations;

CREATE INDEX idx_conversations_updated
ON conversations(archived_at, updated_at DESC);

CREATE TABLE profile_events (
    id TEXT PRIMARY KEY,
    user_profile_id TEXT NOT NULL REFERENCES user_profiles(id),
    item_kind TEXT NOT NULL CHECK(length(trim(item_kind)) > 0),
    item_name TEXT NOT NULL CHECK(length(trim(item_name)) > 0),
    value TEXT NOT NULL CHECK(length(trim(value)) > 0),
    origin TEXT NOT NULL CHECK(origin IN (
        'user_asserted', 'ai_auto_saved', 'ai_proposed'
    )),
    approval TEXT NOT NULL CHECK(approval IN (
        'auto_saved', 'confirmed', 'pending_confirmation'
    )),
    scope TEXT NOT NULL CHECK(scope IN (
        'profile_only', 'continuity', 'selected_characters', 'continuity_cast'
    )),
    scope_count INTEGER NOT NULL CHECK(scope_count BETWEEN 0 AND 5),
    source_conversation_id TEXT REFERENCES conversations(id),
    source_branch_id TEXT,
    source_message_id TEXT,
    manual_operation_id TEXT,
    supersedes_event_id TEXT REFERENCES profile_events(id),
    effective_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    CHECK(source_message_id IS NULL OR source_conversation_id IS NOT NULL),
    CHECK(source_branch_id IS NULL OR source_conversation_id IS NOT NULL),
    CHECK(source_message_id IS NOT NULL OR manual_operation_id IS NOT NULL),
    CHECK(supersedes_event_id IS NULL OR supersedes_event_id <> id),
    FOREIGN KEY(source_branch_id, source_conversation_id)
        REFERENCES branches(id, conversation_id),
    FOREIGN KEY(source_message_id, source_conversation_id)
        REFERENCES messages(id, conversation_id)
);

CREATE TABLE profile_event_scopes (
    event_id TEXT NOT NULL REFERENCES profile_events(id),
    character_id TEXT NOT NULL REFERENCES characters(id),
    PRIMARY KEY(event_id, character_id)
);

CREATE TABLE profile_decisions (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_profile_id TEXT NOT NULL REFERENCES user_profiles(id),
    target_event_id TEXT NOT NULL REFERENCES profile_events(id),
    state TEXT NOT NULL CHECK(state IN (
        'confirmed', 'rejected', 'undone', 'disabled', 'active'
    )),
    actor TEXT NOT NULL CHECK(actor IN ('user', 'ai', 'system')),
    recorded_at TEXT NOT NULL
);

CREATE TABLE profile_capture_suppressions (
    user_profile_id TEXT NOT NULL REFERENCES user_profiles(id),
    source_message_id TEXT NOT NULL REFERENCES messages(id),
    item_kind TEXT NOT NULL,
    item_name TEXT NOT NULL,
    PRIMARY KEY(user_profile_id, source_message_id, item_kind, item_name)
);

CREATE TABLE profile_purge_receipts (
    request_id TEXT PRIMARY KEY,
    user_profile_id TEXT NOT NULL REFERENCES user_profiles(id),
    deleted_event_count INTEGER NOT NULL CHECK(deleted_event_count >= 0),
    deleted_relationship_event_count INTEGER NOT NULL
        CHECK(deleted_relationship_event_count >= 0),
    completed_at TEXT NOT NULL
);

CREATE TABLE profile_purge_authorizations (
    request_id TEXT PRIMARY KEY,
    user_profile_id TEXT NOT NULL REFERENCES user_profiles(id)
);

CREATE TABLE profile_derived_data (
    id TEXT PRIMARY KEY,
    user_profile_id TEXT NOT NULL REFERENCES user_profiles(id),
    kind TEXT NOT NULL CHECK(kind IN ('summary', 'embedding', 'search_index')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE relationship_definitions (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    group_name TEXT NOT NULL,
    display_name TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('symmetric', 'directed')),
    role_a TEXT,
    role_b TEXT,
    caution_tags_json TEXT NOT NULL DEFAULT '[]',
    archived_at TEXT,
    CHECK(
        (direction = 'symmetric' AND role_a IS NULL AND role_b IS NULL)
        OR
        (direction = 'directed' AND length(trim(role_a)) > 0
         AND length(trim(role_b)) > 0)
    )
);

INSERT INTO relationship_definitions(
    id, category, group_name, display_name, direction, role_a, role_b,
    caution_tags_json
) VALUES
    ('friend', '王道・絆', '共闘・信頼', '友人', 'symmetric', NULL, NULL, '[]'),
    ('best-friend', '王道・絆', '共闘・信頼', '親友・盟友', 'symmetric', NULL, NULL, '[]'),
    ('partner', '王道・絆', '共闘・信頼', '相棒（バディ）', 'symmetric', NULL, NULL, '[]'),
    ('rival', '対立・歪み', '宿命の対峙', 'ライバル', 'symmetric', NULL, NULL, '[]'),
    ('mentor', '王道・絆', '育成・継承', '師弟', 'directed', '師', '弟子', '["power_imbalance"]'),
    ('senior', '王道・絆', '憧れ・敬愛', '先輩・後輩', 'directed', '先輩', '後輩', '["power_imbalance"]'),
    ('supervisor', '現実・社会', 'ビジネス', '上司・部下', 'directed', '上司', '部下', '["power_imbalance"]'),
    ('spouses', '現実・社会', '血縁・婚姻', '夫婦', 'symmetric', NULL, NULL, '["sexual_or_romantic"]'),
    ('captor', '対立・歪み', '奪取・支配', '監禁と被監禁', 'directed', '監禁側', '被監禁側', '["coercion_or_confinement","harm_history"]'),
    ('victim-offender', '現実・社会', '社会・日常', '加害者・被害者', 'directed', '加害者', '被害者', '["harm_history"]'),
    ('creator-created', '特殊・現代', '特殊世界観', '創造主と被造物', 'directed', '創造主', '被造物', '["power_imbalance"]'),
    ('memory-gap-friends', '変化球・日常・アウトロー', '記憶・認知', '「私」を知らない幼馴染', 'directed', '覚えている側', '知らない側', '["identity_or_memory_gap"]'),
    ('childhood-friends', '王道・絆', '共闘・信頼', '幼馴染', 'symmetric', NULL, NULL, '[]'),
    ('cohort', '王道・絆', '共闘・信頼', '同期', 'symmetric', NULL, NULL, '[]'),
    ('mentor-mentee', '王道・絆', '育成・継承', 'メンターとメンティ', 'directed', 'メンター', 'メンティ', '["power_imbalance"]'),
    ('rescuer-foundling', '王道・絆', '育成・継承', '拾った側と拾われた側', 'directed', '拾った側', '拾われた側', '["power_imbalance"]'),
    ('king-knight', '王道・絆', '憧れ・敬愛', '王と騎士', 'directed', '王', '騎士', '["power_imbalance"]'),
    ('master-servant', '王道・絆', '憧れ・敬愛', '主従関係', 'directed', '主人', '従者', '["power_imbalance"]'),
    ('nemesis', '対立・歪み', '宿命の対峙', '宿敵（ネメシス）', 'symmetric', NULL, NULL, '["harm_history"]'),
    ('fateful-ties', '対立・歪み', '宿命の対峙', '因縁', 'symmetric', NULL, NULL, '["harm_history"]'),
    ('light-shadow', '対立・歪み', '宿命の対峙', '光と影', 'directed', '光', '影', '[]'),
    ('affair', '対立・歪み', '背徳・不義', '浮気・不倫', 'symmetric', NULL, NULL, '["sexual_or_romantic","harm_history"]'),
    ('double-affair', '対立・歪み', '背徳・不義', 'W不倫', 'symmetric', NULL, NULL, '["sexual_or_romantic","harm_history"]'),
    ('stolen-love', '対立・歪み', '背徳・不義', '略奪愛', 'directed', '奪う側', '奪われる側', '["sexual_or_romantic","harm_history"]'),
    ('friends-with-benefits', '対立・歪み', '背徳・不義', 'セフレ', 'symmetric', NULL, NULL, '["sexual_or_romantic"]'),
    ('codependent', '対立・歪み', '愛憎・執着', '共依存', 'symmetric', NULL, NULL, '["harm_history"]'),
    ('overwhelming-feelings', '対立・歪み', '愛憎・執着', 'クソデカ感情', 'directed', '感情を向ける側', '向けられる側', '[]'),
    ('worship', '対立・歪み', '愛憎・執着', '崇拝', 'directed', '崇拝する側', '崇拝される側', '["power_imbalance"]'),
    ('love-triangle', '対立・歪み', '愛憎・執着', '泥沼の三角関係', 'symmetric', NULL, NULL, '["sexual_or_romantic","harm_history"]'),
    ('ntr', '対立・歪み', '奪取・支配', '寝取られ（NTR）', 'directed', '奪われる側', '奪う側', '["sexual_or_romantic","harm_history"]'),
    ('ntl', '対立・歪み', '奪取・支配', '寝取り（NTL）', 'directed', '奪う側', '奪われる側', '["sexual_or_romantic","harm_history"]'),
    ('parent-child', '現実・社会', '血縁・婚姻', '親子', 'directed', '親', '子', '["power_imbalance"]'),
    ('siblings', '現実・社会', '血縁・婚姻', '兄弟姉妹', 'symmetric', NULL, NULL, '[]'),
    ('in-laws', '現実・社会', '血縁・婚姻', '義理の家族（姻族）', 'symmetric', NULL, NULL, '[]'),
    ('colleagues', '現実・社会', 'ビジネス', '同僚', 'symmetric', NULL, NULL, '[]'),
    ('client-vendor', '現実・社会', 'ビジネス', 'クライアントとベンダー', 'directed', 'クライアント', 'ベンダー', '["power_imbalance"]'),
    ('neighbors', '現実・社会', '社会・日常', '隣人', 'symmetric', NULL, NULL, '[]'),
    ('shopkeeper-customer', '現実・社会', '社会・日常', '店主と客', 'directed', '店主', '客', '[]'),
    ('plaintiff-defendant', '現実・社会', '社会・日常', '原告・被告', 'directed', '原告', '被告', '["harm_history"]'),
    ('contractor-demon', '特殊・現代', '特殊世界観', '契約者と悪魔', 'directed', '契約者', '悪魔', '["power_imbalance"]'),
    ('interspecies-duo', '特殊・現代', '特殊世界観', '異種族コンビ', 'symmetric', NULL, NULL, '[]'),
    ('queerplatonic', '特殊・現代', 'ネット・現代', 'クィアプラトニック', 'symmetric', NULL, NULL, '[]'),
    ('streamer-listener', '特殊・現代', 'ネット・現代', '配信者とリスナー', 'directed', '配信者', 'リスナー', '["power_imbalance"]'),
    ('online-game-partners', '特殊・現代', 'ネット・現代', 'ネトゲの相方', 'symmetric', NULL, NULL, '[]'),
    ('mutual-unrequited-love', '特殊・現代', 'ギャップ・変化', '両片想い', 'symmetric', NULL, NULL, '["sexual_or_romantic"]'),
    ('fake-lovers', '特殊・現代', 'ギャップ・変化', '偽装恋人', 'symmetric', NULL, NULL, '["sexual_or_romantic"]'),
    ('former-slave-master', '特殊・現代', 'ギャップ・変化', '元奴隷と元主人', 'directed', '元奴隷', '元主人', '["power_imbalance","harm_history"]'),
    ('hostage-guard', '主従・契約・心理', '主従・契約', '人質と監視役', 'directed', '人質', '監視役', '["coercion_or_confinement","power_imbalance"]'),
    ('prisoner-warden', '主従・契約・心理', '主従・契約', '囚人と番人', 'directed', '囚人', '番人', '["coercion_or_confinement","power_imbalance"]'),
    ('god-sacrifice', '主従・契約・心理', '主従・契約', '神と生贄', 'directed', '神', '生贄', '["coercion_or_confinement","power_imbalance"]'),
    ('living-god-priest', '主従・契約・心理', '主従・契約', '生き神と神職', 'directed', '生き神', '神職', '["power_imbalance"]'),
    ('con-artist-target', '主従・契約・心理', '心理・裏切り', '詐欺師とターゲット', 'directed', '詐欺師', 'ターゲット', '["harm_history"]'),
    ('accomplices', '主従・契約・心理', '心理・裏切り', '共犯関係', 'symmetric', NULL, NULL, '["harm_history"]'),
    ('victim-offender-family', '主従・契約・心理', '心理・裏切り', '被害者と加害者の家族', 'directed', '被害者', '加害者の家族', '["harm_history"]'),
    ('host-parasite', 'ファンタジー・特殊設定', '特殊共有', '憑依者（ホスト）と寄生者', 'directed', 'ホスト', '寄生者', '["coercion_or_confinement"]'),
    ('shared-personality', 'ファンタジー・特殊設定', '特殊共有', '人格共有', 'symmetric', NULL, NULL, '["identity_or_memory_gap"]'),
    ('reincarnate-old-acquaintance', 'ファンタジー・特殊設定', '特殊共有', '転生者と元の世界の知人', 'directed', '転生者', '元の世界の知人', '["identity_or_memory_gap"]'),
    ('author-character', 'ファンタジー・特殊設定', '特殊共有', '原作者とキャラクター', 'directed', '原作者', 'キャラクター', '["power_imbalance"]'),
    ('prophet-hero', 'ファンタジー・特殊設定', '神話・運命', '未来を知る者と抗う者（預言者と英雄）', 'directed', '預言者', '英雄', '["identity_or_memory_gap"]'),
    ('reincarnation-pair', 'ファンタジー・特殊設定', '神話・運命', '幾度も輪廻転生を繰り返す者同士', 'symmetric', NULL, NULL, '["identity_or_memory_gap"]'),
    ('former-rivals-cofounders', '変化球・日常・アウトロー', '変化・再起', '元ライバル同士の共同経営者', 'symmetric', NULL, NULL, '["harm_history"]'),
    ('retired-hero-former-villain', '変化球・日常・アウトロー', '変化・再起', '引退ヒーローと元悪の幹部', 'directed', '引退ヒーロー', '元悪の幹部', '["harm_history"]'),
    ('amnesiac-lovers', '変化球・日常・アウトロー', '記憶・認知', '記憶を失った恋人', 'directed', '覚えている恋人', '記憶を失った恋人', '["identity_or_memory_gap","sexual_or_romantic"]'),
    ('body-swap-parties', '変化球・日常・アウトロー', '記憶・認知', '入れ替わり当事者', 'symmetric', NULL, NULL, '["identity_or_memory_gap"]'),
    ('fake-family', '変化球・日常・アウトロー', '疑似家族・逃避', '偽りの家族', 'symmetric', NULL, NULL, '[]'),
    ('mission-spouses', '変化球・日常・アウトロー', '疑似家族・逃避', '任務のための夫婦', 'symmetric', NULL, NULL, '["sexual_or_romantic"]'),
    ('fugitive-pursuer', '変化球・日常・アウトロー', '疑似家族・逃避', '逃亡者と追跡者', 'directed', '逃亡者', '追跡者', '["harm_history"]'),
    ('underworld-traitors', '変化球・日常・アウトロー', '疑似家族・逃避', '闇社会の裏切り者同士', 'symmetric', NULL, NULL, '["harm_history"]');

CREATE TABLE relationship_events (
    id TEXT PRIMARY KEY,
    continuity_id TEXT NOT NULL,
    user_profile_id TEXT NOT NULL,
    character_id TEXT NOT NULL REFERENCES characters(id),
    source_conversation_id TEXT,
    source_branch_id TEXT,
    source_message_id TEXT,
    meaning TEXT NOT NULL CHECK(meaning IN (
        'positive_interaction', 'kept_commitment', 'respected_boundary',
        'conflict', 'boundary_violation', 'repeated_boundary_violation',
        'repair', 'relationship_set', 'relationship_retired', 'reset'
    )),
    severity TEXT NOT NULL CHECK(severity IN ('low', 'medium', 'high')),
    evidence_context TEXT NOT NULL CHECK(evidence_context IN (
        'direct', 'quoted', 'hypothetical', 'roleplay', 'narrative',
        'third_party', 'unknown'
    )),
    evidence_start INTEGER,
    evidence_end INTEGER,
    reason TEXT NOT NULL CHECK(length(trim(reason)) > 0),
    approval TEXT NOT NULL CHECK(approval IN (
        'auto_applied', 'confirmed', 'pending_confirmation'
    )),
    policy_version TEXT NOT NULL,
    knowledge_count INTEGER NOT NULL CHECK(knowledge_count BETWEEN 0 AND 5),
    relationship_definition_id TEXT REFERENCES relationship_definitions(id),
    assignment_state TEXT CHECK(assignment_state IN (
        'proposed', 'active', 'historical', 'disabled'
    )),
    role TEXT,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY(continuity_id, user_profile_id)
        REFERENCES continuities(id, user_profile_id),
    FOREIGN KEY(source_conversation_id, continuity_id)
        REFERENCES conversations(id, continuity_id),
    FOREIGN KEY(source_branch_id, source_conversation_id)
        REFERENCES branches(id, conversation_id),
    FOREIGN KEY(source_message_id, source_conversation_id)
        REFERENCES messages(id, conversation_id),
    CHECK(
        (source_conversation_id IS NULL AND source_branch_id IS NULL
         AND source_message_id IS NULL)
        OR
        (source_conversation_id IS NOT NULL AND source_branch_id IS NOT NULL
         AND source_message_id IS NOT NULL)
    ),
    CHECK(
        (evidence_start IS NULL AND evidence_end IS NULL)
        OR
        (evidence_start >= 0 AND evidence_end > evidence_start)
    ),
    CHECK(
        (meaning IN ('relationship_set', 'relationship_retired')
         AND relationship_definition_id IS NOT NULL
         AND assignment_state IS NOT NULL)
        OR
        (meaning NOT IN ('relationship_set', 'relationship_retired')
         AND relationship_definition_id IS NULL
         AND assignment_state IS NULL
         AND role IS NULL)
    )
);

CREATE TABLE relationship_event_knowledge (
    event_id TEXT NOT NULL REFERENCES relationship_events(id),
    character_id TEXT NOT NULL REFERENCES characters(id),
    PRIMARY KEY(event_id, character_id)
);

CREATE TABLE relationship_decisions (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_profile_id TEXT NOT NULL REFERENCES user_profiles(id),
    target_event_id TEXT NOT NULL REFERENCES relationship_events(id),
    state TEXT NOT NULL CHECK(state IN ('confirmed', 'rejected', 'undone')),
    actor TEXT NOT NULL CHECK(actor IN ('user', 'ai', 'system')),
    recorded_at TEXT NOT NULL
);

CREATE TABLE relationship_interpretations (
    id TEXT PRIMARY KEY,
    continuity_id TEXT NOT NULL,
    user_profile_id TEXT NOT NULL,
    character_id TEXT NOT NULL REFERENCES characters(id),
    character_version_id TEXT NOT NULL REFERENCES character_versions(id),
    relationship_definition_ids_json TEXT NOT NULL,
    summary TEXT NOT NULL CHECK(length(trim(summary)) > 0),
    evidence_event_ids_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN (
        'current', 'superseded', 'invalidated', 'recomputing'
    )),
    generated_at TEXT NOT NULL,
    FOREIGN KEY(continuity_id, user_profile_id)
        REFERENCES continuities(id, user_profile_id)
);

CREATE UNIQUE INDEX uq_relationship_current_interpretation
ON relationship_interpretations(continuity_id, user_profile_id, character_id)
WHERE state = 'current';

CREATE INDEX idx_profile_events_profile
ON profile_events(user_profile_id, item_kind, item_name, recorded_at, id);

CREATE INDEX idx_profile_decisions_target
ON profile_decisions(target_event_id, sequence);

CREATE INDEX idx_relationship_events_ledger
ON relationship_events(
    continuity_id, user_profile_id, character_id, recorded_at, id
);

CREATE INDEX idx_relationship_decisions_target
ON relationship_decisions(target_event_id, sequence);

CREATE TRIGGER profile_events_no_update
BEFORE UPDATE ON profile_events
BEGIN
    SELECT RAISE(ABORT, 'profile events are append-only');
END;

CREATE TRIGGER profile_events_guard_ai_supersede
BEFORE INSERT ON profile_events
WHEN NEW.supersedes_event_id IS NOT NULL
 AND NEW.origin <> 'user_asserted'
 AND (
    SELECT origin FROM profile_events WHERE id = NEW.supersedes_event_id
 ) = 'user_asserted'
BEGIN
    SELECT RAISE(ABORT, 'AI cannot supersede user asserted profile');
END;

CREATE TRIGGER profile_events_guard_suppression
BEFORE INSERT ON profile_events
WHEN NEW.source_message_id IS NOT NULL
 AND EXISTS (
    SELECT 1 FROM profile_capture_suppressions suppression
    WHERE suppression.user_profile_id = NEW.user_profile_id
      AND suppression.source_message_id = NEW.source_message_id
      AND suppression.item_kind = NEW.item_kind
      AND suppression.item_name = NEW.item_name
 )
BEGIN
    SELECT RAISE(ABORT, 'purged profile source is suppressed');
END;

CREATE TRIGGER profile_events_no_delete
BEFORE DELETE ON profile_events
WHEN NOT EXISTS (
    SELECT 1 FROM profile_purge_authorizations authorization
    WHERE authorization.user_profile_id = OLD.user_profile_id
)
BEGIN
    SELECT RAISE(ABORT, 'profile events require an authorized purge');
END;

CREATE TRIGGER profile_scope_capacity
BEFORE INSERT ON profile_event_scopes
WHEN (
    SELECT COUNT(*) FROM profile_event_scopes WHERE event_id = NEW.event_id
) >= (
    SELECT scope_count FROM profile_events WHERE id = NEW.event_id
)
BEGIN
    SELECT RAISE(ABORT, 'profile scope capacity exceeded');
END;

CREATE TRIGGER profile_scopes_no_update
BEFORE UPDATE ON profile_event_scopes
BEGIN
    SELECT RAISE(ABORT, 'profile scopes are append-only');
END;

CREATE TRIGGER profile_scopes_no_delete
BEFORE DELETE ON profile_event_scopes
WHEN NOT EXISTS (
    SELECT 1
    FROM profile_events event
    JOIN profile_purge_authorizations authorization
      ON authorization.user_profile_id = event.user_profile_id
    WHERE event.id = OLD.event_id
)
BEGIN
    SELECT RAISE(ABORT, 'profile scopes require an authorized purge');
END;

CREATE TRIGGER profile_decisions_no_update
BEFORE UPDATE ON profile_decisions
BEGIN
    SELECT RAISE(ABORT, 'profile decisions are append-only');
END;

CREATE TRIGGER profile_decisions_owner_guard
BEFORE INSERT ON profile_decisions
WHEN NEW.user_profile_id <> (
    SELECT user_profile_id FROM profile_events WHERE id = NEW.target_event_id
)
BEGIN
    SELECT RAISE(ABORT, 'profile decision owner mismatch');
END;

CREATE TRIGGER profile_decisions_actor_guard
BEFORE INSERT ON profile_decisions
WHEN NEW.actor <> 'user' AND NEW.state IN (
    'confirmed', 'rejected', 'disabled', 'active'
)
BEGIN
    SELECT RAISE(ABORT, 'profile decision requires the user');
END;

CREATE TRIGGER profile_decisions_no_delete
BEFORE DELETE ON profile_decisions
WHEN NOT EXISTS (
    SELECT 1 FROM profile_purge_authorizations authorization
    WHERE authorization.user_profile_id = OLD.user_profile_id
)
BEGIN
    SELECT RAISE(ABORT, 'profile decisions require an authorized purge');
END;

CREATE TRIGGER relationship_events_no_update
BEFORE UPDATE ON relationship_events
BEGIN
    SELECT RAISE(ABORT, 'relationship events are append-only');
END;

CREATE TRIGGER relationship_events_no_delete
BEFORE DELETE ON relationship_events
WHEN NOT EXISTS (
    SELECT 1 FROM profile_purge_authorizations authorization
    WHERE authorization.user_profile_id = OLD.user_profile_id
)
BEGIN
    SELECT RAISE(ABORT, 'relationship events require an authorized purge');
END;

CREATE TRIGGER relationship_knowledge_capacity
BEFORE INSERT ON relationship_event_knowledge
WHEN (
    SELECT COUNT(*) FROM relationship_event_knowledge WHERE event_id = NEW.event_id
) >= (
    SELECT knowledge_count FROM relationship_events WHERE id = NEW.event_id
)
BEGIN
    SELECT RAISE(ABORT, 'relationship knowledge capacity exceeded');
END;

CREATE TRIGGER relationship_knowledge_no_update
BEFORE UPDATE ON relationship_event_knowledge
BEGIN
    SELECT RAISE(ABORT, 'relationship knowledge is append-only');
END;

CREATE TRIGGER relationship_knowledge_no_delete
BEFORE DELETE ON relationship_event_knowledge
WHEN NOT EXISTS (
    SELECT 1
    FROM relationship_events event
    JOIN profile_purge_authorizations authorization
      ON authorization.user_profile_id = event.user_profile_id
    WHERE event.id = OLD.event_id
)
BEGIN
    SELECT RAISE(ABORT, 'relationship knowledge requires an authorized purge');
END;

CREATE TRIGGER relationship_decisions_no_update
BEFORE UPDATE ON relationship_decisions
BEGIN
    SELECT RAISE(ABORT, 'relationship decisions are append-only');
END;

CREATE TRIGGER relationship_decisions_owner_guard
BEFORE INSERT ON relationship_decisions
WHEN NEW.user_profile_id <> (
    SELECT user_profile_id FROM relationship_events WHERE id = NEW.target_event_id
)
BEGIN
    SELECT RAISE(ABORT, 'relationship decision owner mismatch');
END;

CREATE TRIGGER relationship_decisions_actor_guard
BEFORE INSERT ON relationship_decisions
WHEN NEW.actor <> 'user'
BEGIN
    SELECT RAISE(ABORT, 'relationship decision requires the user');
END;

CREATE TRIGGER relationship_decisions_no_delete
BEFORE DELETE ON relationship_decisions
WHEN NOT EXISTS (
    SELECT 1 FROM profile_purge_authorizations authorization
    WHERE authorization.user_profile_id = OLD.user_profile_id
)
BEGIN
    SELECT RAISE(ABORT, 'relationship decisions require an authorized purge');
END;

PRAGMA foreign_keys = ON;
