import {Pool} from 'pg';
import logger from '~/shared/utils/logger';
import {CHAT_TARGET_TYPE} from '~/shared/models/ChatModel';

const ANALYTICS_ENABLED = Boolean(process.env.ANALYTICS_PG_HOST);
const SUMMARY_WINDOW_DAYS = Number(process.env.ANALYTICS_SUMMARY_DAYS || 30);

let pool = null;
let ready = Promise.resolve();

const safeGet = (record, path) => record && typeof record.getIn === 'function'
  ? record.getIn(path)
  : null;

const normalizeImmutable = (value) => {
  if (!value) return null;
  if (typeof value.toJS === 'function') return value.toJS();
  if (typeof value.toObject === 'function') return value.toObject();
  return value;
};

const truncateString = (value, length = 512) => {
  if (typeof value !== 'string') return value;
  return value.length > length ? `${value.slice(0, length)}…` : value;
};

const sanitizeContext = (value, depth = 0) => {
  if (value == null) return value;
  if (depth > 3) return typeof value === 'object' ? '[Object]' : value;
  if (Array.isArray(value)) {
    return value.slice(0, 20).map(item => sanitizeContext(item, depth + 1));
  }
  if (typeof value === 'object') {
    const result = {};
    Object.keys(value).slice(0, 20).forEach((key) => {
      const sanitizedValue = sanitizeContext(value[key], depth + 1);
      if (sanitizedValue !== undefined) {
        result[key] = truncateString(sanitizedValue);
      }
    });
    return result;
  }
  if (typeof value === 'string') {
    return truncateString(value);
  }
  return value;
};

const normalizeUser = (userRecord) => {
  if (!userRecord) return null;
  const plain = normalizeImmutable(userRecord);
  if (!plain || !plain.id) return null;
  const profile = plain.profile ? normalizeImmutable(plain.profile) : {};
  return {
    externalId: plain.id,
    login: plain.login,
    authType: plain.authType || null,
    vkLogin: profile && (profile.vkLogin || profile.authName || null),
    vkId: profile && (profile.vkId || null)
  };
};

const initializeSchema = async (currentPool) => {
  await currentPool.query(`
    CREATE TABLE IF NOT EXISTS analytics_users (
      id SERIAL PRIMARY KEY,
      external_id TEXT UNIQUE NOT NULL,
      login TEXT,
      auth_type TEXT,
      vk_login TEXT,
      vk_id TEXT,
      created_at TIMESTAMPTZ DEFAULT now(),
      updated_at TIMESTAMPTZ DEFAULT now()
    );
  `);

  await currentPool.query(`
    CREATE TABLE IF NOT EXISTS analytics_events (
      id BIGSERIAL PRIMARY KEY,
      external_user_id TEXT,
      user_id INTEGER REFERENCES analytics_users(id) ON DELETE SET NULL,
      event_type TEXT NOT NULL,
      event_context JSONB DEFAULT '{}'::jsonb,
      room_id TEXT,
      game_id TEXT,
      created_at TIMESTAMPTZ DEFAULT now()
    );
  `);

  await currentPool.query('CREATE INDEX IF NOT EXISTS analytics_events_event_type_idx ON analytics_events(event_type);');
  await currentPool.query('CREATE INDEX IF NOT EXISTS analytics_events_created_at_idx ON analytics_events(created_at);');
  await currentPool.query('CREATE INDEX IF NOT EXISTS analytics_events_game_id_idx ON analytics_events(game_id);');
  await currentPool.query('CREATE INDEX IF NOT EXISTS analytics_events_room_id_idx ON analytics_events(room_id);');
  await currentPool.query('CREATE INDEX IF NOT EXISTS analytics_events_game_event_idx ON analytics_events(game_id, event_type);');
};

const ensurePool = () => {
  if (!ANALYTICS_ENABLED) return null;
  if (pool) return pool;

  pool = new Pool({
    host: process.env.ANALYTICS_PG_HOST,
    port: Number(process.env.ANALYTICS_PG_PORT || 5432),
    database: process.env.ANALYTICS_PG_DATABASE,
    user: process.env.ANALYTICS_PG_USER,
    password: process.env.ANALYTICS_PG_PASSWORD,
    max: Number(process.env.ANALYTICS_PG_POOL_SIZE || 10),
    ssl: process.env.ANALYTICS_PG_SSL === 'true'
  });

  pool.on('error', (error) => {
    logger.error('Analytics pool error', error);
  });

  ready = (async () => {
    try {
      await initializeSchema(pool);
    } catch (error) {
      logger.error('Failed to initialize analytics storage', error);
      throw error;
    }
  })();

  return pool;
};

const ensureUserRecord = async (userData) => {
  if (!ANALYTICS_ENABLED || !userData || !userData.externalId) return {id: null, ...userData};
  const currentPool = ensurePool();
  const values = [
    userData.externalId,
    userData.login || null,
    userData.authType || null,
    userData.vkLogin || null,
    userData.vkId || null
  ];
  await ready;

  const {rows} = await currentPool.query(`
    INSERT INTO analytics_users (external_id, login, auth_type, vk_login, vk_id)
    VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT (external_id)
    DO UPDATE SET
      login = EXCLUDED.login,
      auth_type = EXCLUDED.auth_type,
      vk_login = EXCLUDED.vk_login,
      vk_id = EXCLUDED.vk_id,
      updated_at = now()
    RETURNING id;
  `, values);
  return {id: rows[0] ? rows[0].id : null, ...userData};
};

const logEvent = async ({eventType, user, metadata = {}, roomId = null, gameId = null}) => {
  if (!ANALYTICS_ENABLED) return;
  const currentPool = ensurePool();
  await ready;
  const normalizedContext = sanitizeContext(metadata) || {};
  let userRecord = user || null;
  if (userRecord && !userRecord.id) {
    userRecord = await ensureUserRecord(userRecord);
  }
  await currentPool.query(`
    INSERT INTO analytics_events (external_user_id, user_id, event_type, event_context, room_id, game_id)
    VALUES ($1, $2, $3, $4::jsonb, $5, $6);
  `, [
    userRecord ? userRecord.externalId : null,
    userRecord ? userRecord.id : null,
    eventType,
    JSON.stringify(normalizedContext),
    roomId,
    gameId
  ]);
};

const summarizeGame = (game) => {
  if (!game) return {};
  const plainGame = normalizeImmutable(game);
  const players = plainGame.players
    ? Object.values(plainGame.players).map(player => ({
      id: player.id,
      index: player.index,
      score: player.score
    }))
    : [];
  return {
    gameId: plainGame.id,
    roomId: plainGame.roomId,
    phase: plainGame.status && plainGame.status.phase,
    round: plainGame.status && plainGame.status.round,
    winnerId: plainGame.winnerId,
    players
  };
};

const handleChatMessage = async (action, prevState) => {
  const messageRecord = action.data && action.data.message;
  if (!messageRecord) return;
  const message = normalizeImmutable(messageRecord);
  const fromUserId = message.from;
  const userRecord = fromUserId && fromUserId !== '0'
    ? normalizeUser(safeGet(prevState, ['users', fromUserId]))
    : null;
  await logEvent({
    eventType: 'chat_message',
    user: userRecord,
    roomId: message.toType === CHAT_TARGET_TYPE.ROOM ? message.to : null,
    metadata: {
      to: message.to,
      toType: message.toType,
      text: message.text,
      context: message.context
    }
  });
};

const extractRoomUser = (data, prevState, nextState) => {
  if (!data) return null;
  const userId = data.userId || data.hostId || data.kickUserId || data.targetUserId || data.banUserId;
  if (!userId) return null;
  return normalizeUser(safeGet(nextState, ['users', userId]) || safeGet(prevState, ['users', userId]));
};

const resolveRoomId = (data) => {
  if (!data) return null;
  if (data.roomId) return data.roomId;
  if (data.room && data.room.id) return data.room.id;
  return null;
};

const handleRoomEvent = async (action, prevState, nextState) => {
  const {type, data} = action;
  switch (type) {
    case 'roomCreate': {
      const room = data && normalizeImmutable(data.room);
      if (!room) return;
      const host = normalizeUser(safeGet(nextState, ['users', room.hostId]) || safeGet(prevState, ['users', room.hostId]));
      await logEvent({
        eventType: 'room_created',
        user: host,
        roomId: room.id,
        metadata: {roomName: room.name}
      });
      return;
    }
    case 'roomJoin': {
      const roomId = data && data.roomId;
      const userId = data && data.userId;
      const user = normalizeUser(safeGet(nextState, ['users', userId]) || safeGet(prevState, ['users', userId]));
      await logEvent({
        eventType: 'room_joined',
        user,
        roomId,
        metadata: {roomId}
      });
      return;
    }
    case 'roomSpectate': {
      const roomId = data && data.roomId;
      const userId = data && data.userId;
      const user = normalizeUser(safeGet(nextState, ['users', userId]) || safeGet(prevState, ['users', userId]));
      await logEvent({
        eventType: 'room_spectated',
        user,
        roomId,
        metadata: {roomId}
      });
      return;
    }
    case 'roomExit': {
      const roomId = data && data.roomId;
      const userId = data && data.userId;
      const user = normalizeUser(safeGet(prevState, ['users', userId]) || safeGet(nextState, ['users', userId]));
      await logEvent({
        eventType: 'room_left',
        user,
        roomId,
        metadata: {roomId}
      });
      return;
    }
    default:
      break;
  }

  if (!type.startsWith('room')) return;
  if (type.endsWith('Request') || type.endsWith('Self')) return;

  const metadata = normalizeImmutable(data) || {};
  await logEvent({
    eventType: type,
    user: extractRoomUser(metadata, prevState, nextState),
    roomId: resolveRoomId(metadata),
    metadata
  });
};

const handleGameEvent = async (action, prevState, nextState) => {
  const {type, data} = action;
  const gameId = data && data.gameId;
  if (!gameId) return;
  const gameState = safeGet(type === 'gameCreateSuccess' ? nextState : prevState, ['games', gameId])
    || safeGet(nextState, ['games', gameId]);
  const gameSummary = summarizeGame(gameState);
  const basePayload = {
    gameId,
    roomId: gameSummary.roomId
  };
  switch (type) {
    case 'gameCreateSuccess': {
      const room = safeGet(nextState, ['rooms', gameSummary.roomId]) || safeGet(prevState, ['rooms', gameSummary.roomId]);
      const hostId = room && room.hostId;
      const host = normalizeUser(safeGet(nextState, ['users', hostId]) || safeGet(prevState, ['users', hostId]));
      await logEvent({
        eventType: 'game_started',
        user: host,
        gameId,
        roomId: gameSummary.roomId,
        metadata: {
          players: gameSummary.players,
          seed: gameState && gameState.settings && gameState.settings.seed
        }
      });
      break;
    }
    case 'gameEnd': {
      await logEvent({
        eventType: 'game_finished',
        gameId,
        roomId: gameSummary.roomId,
        user: null,
        metadata: gameSummary
      });
      break;
    }
    case 'gameEndTurn':
    case 'gameStartTurn':
    case 'gameStartPhase':
    case 'playerActed':
    case 'gameSetUserTimedOut':
    case 'gameGiveCards': {
      const userId = data && (data.userId || data.playerId);
      const user = normalizeUser(safeGet(prevState, ['users', userId]) || safeGet(nextState, ['users', userId]));
      const metadata = normalizeImmutable(data) || {};
      if (metadata.cards) {
        const cards = metadata.cards;
        const cardCount = Array.isArray(cards)
          ? cards.length
          : (cards && typeof cards.size === 'number') ? cards.size : undefined;
        metadata.cardCount = cardCount;
        delete metadata.cards;
      }
      await logEvent({
        eventType: type,
        user,
        ...basePayload,
        metadata
      });
      break;
    }
    default: {
      if (type.startsWith('game') || type.startsWith('trait')) {
        const metadata = normalizeImmutable(data) || {};
        await logEvent({
          eventType: type,
          ...basePayload,
          metadata
        });
      }
    }
  }
};

const handleAuthenticationEvent = async (action, prevState) => {
  const {type, data} = action;
  switch (type) {
    case 'loginUser': {
      const user = normalizeUser(data && data.user);
      if (!user) return;
      const ensured = await ensureUserRecord(user);
      await logEvent({
        eventType: 'user_login',
        user: ensured,
        metadata: {authType: ensured.authType, login: ensured.login}
      });
      break;
    }
    case 'logoutUser': {
      const userId = data && data.userId;
      const user = normalizeUser(safeGet(prevState, ['users', userId]));
      await logEvent({
        eventType: 'user_logout',
        user,
        metadata: {userId}
      });
      break;
    }
    default:
      break;
  }
};

const AUTH_ACTIONS = new Set(['loginUser', 'logoutUser']);
const CHAT_ACTIONS = new Set(['chatMessageGlobal', 'chatMessageRoom', 'chatMessageUser']);
const ROOM_ACTION_PATTERN = /^room(?:$|[A-Z:_])/;
const GAME_ACTION_PATTERN = /^(?:game|trait)(?:$|[A-Z:_])/;

const handleAction = async (action, prevState, nextState) => {
  if (!ANALYTICS_ENABLED) return;
  if (!action || (action.meta && action.meta.clientOnly)) return;
  const {type} = action;
  if (!type) return;
  try {
    if (AUTH_ACTIONS.has(type)) {
      await handleAuthenticationEvent(action, prevState, nextState);
      return;
    }

    if (CHAT_ACTIONS.has(type)) {
      await handleChatMessage(action, prevState, nextState);
      return;
    }

    if (ROOM_ACTION_PATTERN.test(type)) {
      await handleRoomEvent(action, prevState, nextState);
      return;
    }

    if (GAME_ACTION_PATTERN.test(type)) {
      await handleGameEvent(action, prevState, nextState);
    }
  } catch (error) {
    logger.error('Failed to log analytics event', error, action && action.type);
  }
};

const fetchSummary = async () => {
  if (!ANALYTICS_ENABLED) {
    return {enabled: false};
  }
  const currentPool = ensurePool();
  await ready;
  const windowClause = SUMMARY_WINDOW_DAYS > 0 ? `WHERE created_at >= now() - interval '${SUMMARY_WINDOW_DAYS} days'` : '';
  const [totals, usersOverTime, gamesOverTime, actionsByType, recentEvents, openGames] = await Promise.all([
    currentPool.query('SELECT COUNT(*)::int AS total_events FROM analytics_events;'),
    currentPool.query(`
      SELECT to_char(date_trunc('day', created_at), 'YYYY-MM-DD') AS day,
             COUNT(DISTINCT external_user_id) AS count
      FROM analytics_events
      ${windowClause}
      GROUP BY day
      ORDER BY day ASC;
    `),
    currentPool.query(`
      SELECT to_char(date_trunc('day', created_at), 'YYYY-MM-DD') AS day,
             COUNT(*) AS count
      FROM analytics_events
      WHERE event_type = 'game_started'
      ${windowClause ? `AND created_at >= now() - interval '${SUMMARY_WINDOW_DAYS} days'` : ''}
      GROUP BY day
      ORDER BY day ASC;
    `),
    currentPool.query(`
      SELECT event_type, COUNT(*) AS count
      FROM analytics_events
      ${windowClause}
      GROUP BY event_type
      ORDER BY COUNT(*) DESC
      LIMIT 25;
    `),
    currentPool.query(`
      SELECT e.event_type,
             e.event_context,
             e.created_at,
             e.room_id,
             e.game_id,
             u.login,
             u.vk_login
      FROM analytics_events e
      LEFT JOIN analytics_users u ON e.user_id = u.id
      ORDER BY e.created_at DESC
      LIMIT 25;
    `),
    currentPool.query(`
      SELECT COUNT(DISTINCT e.game_id) AS games_in_progress
      FROM analytics_events e
      WHERE e.event_type = 'game_started'
        AND NOT EXISTS (
          SELECT 1 FROM analytics_events f
          WHERE f.game_id = e.game_id AND f.event_type = 'game_finished'
        );
    `)
  ]);

  const activeUsers = await currentPool.query(`
    SELECT COUNT(DISTINCT external_user_id)::int AS active_users
    FROM analytics_events
    WHERE created_at >= now() - interval '7 days';
  `);

  return {
    enabled: true,
    totals: {
      events: totals.rows[0] ? Number(totals.rows[0].total_events) : 0,
      activeUsers7d: activeUsers.rows[0] ? Number(activeUsers.rows[0].active_users) : 0,
      gamesInProgress: openGames.rows[0] ? Number(openGames.rows[0].games_in_progress) : 0
    },
    usersOverTime: usersOverTime.rows,
    gamesOverTime: gamesOverTime.rows,
    actionsByType: actionsByType.rows,
    recentEvents: recentEvents.rows.map(row => ({
      eventType: row.event_type,
      createdAt: row.created_at,
      roomId: row.room_id,
      gameId: row.game_id,
      login: row.login,
      vkLogin: row.vk_login,
      context: row.event_context
    }))
  };
};

export default {
  get enabled() {
    return ANALYTICS_ENABLED;
  },
  get ready() {
    ensurePool();
    return ready;
  },
  ensureUserRecord,
  logEvent,
  handleAction,
  fetchSummary
};
