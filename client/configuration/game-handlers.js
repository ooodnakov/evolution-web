import {appPlaySound} from '../actions/app';
import {redirectTo} from '../../shared/utils/history';
import {gameServerToClient} from '../../shared/actions/game';

/**
 * Browser-only side effects for otherwise shared server->client game handlers.
 *
 * `shared/actions/game.js` keeps pure redux mapping; routing and sound live
 * here, on top of the shared registry.
 */
export const gameServerToClientOverrides = {
  gameInit: (data, currentUserId) => (dispatch) => {
    dispatch(gameServerToClient.gameInit(data, currentUserId));
    redirectTo('/room');
  }
  , gameCreateSuccess: (data, currentUserId) => (dispatch) => {
    dispatch(gameServerToClient.gameCreateSuccess(data, currentUserId));
    redirectTo('/room');
  }
  , gameNextPlayer: (data, currentUserId) => (dispatch, getState) => {
    const game = getState().get('game');
    const previousPlayerId = game.status.currentPlayer;
    if (previousPlayerId !== currentUserId && data.playerId === currentUserId) dispatch(appPlaySound('NOTIFICATION'));
    dispatch(gameServerToClient.gameNextPlayer(data, currentUserId));
  }
};
