import {db$gameEnd} from '../actions/db';
import {selectGame} from '../../shared/selectors';
import {server$roomSelfDestructStart} from '../../shared/actions/rooms';
import {loggerOnline} from '../../shared/utils/logger';

/**
 * Web-only game lifecycle effects.
 *
 * `shared/actions/game.js` stays free of persistence/socket/room side effects:
 * `server$gameEnd` dispatches authoritative `gameEnd`/client `gameEnd` and then
 * a plain `gameFinished` event. This middleware consumes that event after the
 * reducer ran, so the state already holds the final game.
 */
export const gameLifecycleMiddleware = store => next => action => {
  const result = next(action);

  if (action && action.type === 'gameFinished') {
    const {gameId, finished} = action.data;
    const getState = store.getState;
    const game = selectGame(getState, gameId);

    store.dispatch(server$roomSelfDestructStart(game.roomId, selectGame(getState, gameId).winnerId));

    const gameStats = selectGame(getState, gameId).toDatabase(getState, finished);
    db$gameEnd(gameStats);

    loggerOnline.info(`Game ${finished ? 'finished' : 'exited'} ${game.players.map(p => getState().getIn(['users', p.id, 'login'])).join(', ')}`);
  }

  return result;
};
