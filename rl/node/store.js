import {Record, Map, List} from 'immutable';
import {createStore, applyMiddleware} from 'redux';
import thunk from 'redux-thunk';

import {reducer as gamesReducer} from '~/server/reducers/games-rdx-server';
import {createScheduler} from './scheduler';

/**
 * Isolated authoritative store for one RL episode.
 *
 * Only the gameplay reducer is reused. Lobby reducers (rooms/users/chat) and
 * the socket middleware are web concerns; the environment keeps a minimal
 * `rooms` slice because `selectUsersInGame` resolves seat data through it.
 *
 * `meta.clientOnly` actions must never reach the reducer: `server$gameCreateSuccess`
 * emits private per-user copies of the authoritative game; applying them would
 * overwrite the real state with a masked one.
 */

export const ENV_ROOM_ID = 'room_env';

export class EnvironmentState extends Record({
  games: Map()
  , rooms: Map()
}) {
}

export const roomCreate = (room) => ({type: 'roomCreate', data: {room}});

/**
 * Minimal room slice: a room exists, carries seats/settings and never records
 * `gameId` from `gameCreateNotify` (otherwise a second reset in the same
 * process would fail `checkRoomIsNotInGame`).
 */
export const roomsReducer = (state = Map(), action = {}) => {
  switch (action.type) {
    case 'roomCreate':
      return state.set(action.data.room.id, action.data.room);
    case 'roomDestroy':
      return state.remove(action.data.roomId);
    default:
      return state;
  }
};

export const environmentReducer = (state = new EnvironmentState(), action = {}) => {
  const games = gamesReducer(state.games, action);
  const rooms = roomsReducer(state.rooms, action);
  if (games === state.games && rooms === state.rooms) {
    return state;
  }
  return state.withMutations((next) => next.set('games', games).set('rooms', rooms));
};

const clientOnlyFilter = () => (next) => (action) => {
  if (action && action.meta && action.meta.clientOnly) {
    return undefined;
  }
  return next(action);
};

/**
 * Keeps non-user-level failures distinguishable for the environment: a thrown
 * ActionCheckError is an expected rejected attempt, everything else is an
 * engine invariant violation. Errors are never swallowed here.
 */
const errorNormalizer = () => (next) => (action) => {
  try {
    return next(action);
  } catch (error) {
    if (error && !error.userLevelError && !error.rlFatal) {
      error.rlFatal = true;
    }
    throw error;
  }
};

export const createEnvironmentStore = () => {
  const scheduler = createScheduler();
  const store = createStore(
    environmentReducer
    , new EnvironmentState()
    , applyMiddleware(
      errorNormalizer
      , thunk
      , clientOnlyFilter
      , scheduler.middleware
    )
  );

  return {store, scheduler};
};

export const getGame = (state, gameId) => state.getIn(['games', gameId]);

export const getRoom = (state, roomId) => state.getIn(['rooms', roomId]);

export const listGameIds = (state) => state.get('games').keySeq().toList();

/** Episode seats only: no chat/host/status/connection data. */
export class EnvironmentRoom extends Record({
  id: null
  , users: List()
  , spectators: List()
  , settings: null
  , gameId: null
}) {
}

export const makeRoom = ({roomId, users, settings}) => new EnvironmentRoom({
  id: roomId
  , users
  , spectators: List()
  , settings
  , gameId: null
});

export default createEnvironmentStore;
