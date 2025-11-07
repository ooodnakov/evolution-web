import analytics from '../analytics';
import logger from '~/shared/utils/logger';

export const analyticsMiddleware = (engine = analytics) => (store) => (next) => (action) => {
  const prevState = store.getState();
  const result = next(action);
  const nextState = store.getState();

  if (!engine || !engine.enabled) {
    return result;
  }

  Promise.resolve(engine.handleAction(action, prevState, nextState))
    .catch((error) => {
      logger.error('Analytics middleware error', error);
    });

  return result;
};

export default analyticsMiddleware;
