/**
 * Deterministic replacement for `shared/utils/reduxTimeout.js`.
 *
 * The web middleware owns native timers. A training episode must never depend
 * on wall-clock time, so this middleware keeps the same control-action
 * semantics but stores deadlines as plain logical state. Nothing fires unless
 * the environment explicitly calls `fire(name)`.
 */

const ADD_TIMEOUT = '@@reduxTimeoutMiddleware/addTimeout';
const CANCEL_TIMEOUT = '@@reduxTimeoutMiddleware/cancelTimeout';
const CHECK_TIMEOUT = '@@reduxTimeoutMiddleware/checkTimeout';
const TEST_SHIFT_TIME = '@@reduxTimeoutMiddleware/testShiftTime';

const isFiniteNumber = (value) => typeof value === 'number' && Number.isFinite(value);

const copyEntry = ({name, remaining, order}) => ({name, remaining, order});

export const createScheduler = () => {
  let entries = new Map();
  let nextOrder = 0;
  let dispatchRef = null;

  const pending = () => Array.from(entries.values())
    .sort((left, right) => left.order - right.order)
    .map(copyEntry);

  const fire = (name) => {
    const entry = entries.get(name);
    if (!entry) {
      return false;
    }
    // Delete before dispatch: a re-entrant fire can never run the callback twice.
    entries.delete(name);
    if (typeof dispatchRef !== 'function') {
      throw new Error('SCHEDULER_NOT_MOUNTED');
    }
    dispatchRef(entry.callback);
    return true;
  };

  const reset = () => {
    entries = new Map();
    nextOrder = 0;
    dispatchRef = null;
  };

  const middleware = ({dispatch}) => {
    dispatchRef = dispatch;
    return (next) => (action) => {
      if (action == null || typeof action !== 'object') {
        return next(action);
      }
      switch (action.type) {
        case ADD_TIMEOUT: {
          const {duration, name, callback} = action.data;
          if (duration === 0) {
            return undefined;
          }
          if (entries.has(name)) {
            throw new Error(`timeout[${name}] already occupied`);
          }
          entries.set(name, {
            name,
            remaining: duration,
            callback,
            order: nextOrder++,
          });
          return undefined;
        }
        case CANCEL_TIMEOUT: {
          const {name} = action.data;
          const entry = entries.get(name);
          if (!entry) {
            return undefined;
          }
          entries.delete(name);
          return entry.remaining;
        }
        case CHECK_TIMEOUT: {
          const {name} = action.data;
          return entries.has(name);
        }
        case TEST_SHIFT_TIME: {
          const {time} = action.data;
          entries.forEach((entry) => {
            entry.remaining = Math.max(0, entry.remaining - time);
          });
          return undefined;
        }
        default:
          return next(action);
      }
    };
  };

  /**
   * Snapshot for the episode's opaque handle. Callback references are kept
   * alongside the plain entries so `restore` can rebuild a live scheduler.
   */
  const capture = () => ({
    entries: pending(),
    callbacks: Array.from(entries.values())
      .sort((left, right) => left.order - right.order)
      .map(({name, callback}) => ({name, callback})),
    nextOrder,
  });

  const restore = (captured) => {
    if (captured == null || typeof captured !== 'object'
      || !Array.isArray(captured.entries)
      || !Array.isArray(captured.callbacks)
      || !isFiniteNumber(captured.nextOrder)) {
      throw new Error('INVALID_SCHEDULER_STATE');
    }
    const callbacks = new Map();
    captured.callbacks.forEach((item) => {
      if (item == null || typeof item.name !== 'string' || typeof item.callback !== 'function') {
        throw new Error('INVALID_SCHEDULER_STATE');
      }
      callbacks.set(item.name, item.callback);
    });
    const restored = new Map();
    captured.entries.forEach((item) => {
      if (item == null
        || typeof item.name !== 'string'
        || !isFiniteNumber(item.remaining)
        || item.remaining < 0
        || !isFiniteNumber(item.order)
        || !callbacks.has(item.name)) {
        throw new Error('INVALID_SCHEDULER_STATE');
      }
      restored.set(item.name, {
        name: item.name,
        remaining: item.remaining,
        callback: callbacks.get(item.name),
        order: item.order,
      });
    });
    entries = restored;
    nextOrder = captured.nextOrder;
  };

  return {
    middleware,
    pending,
    capture,
    restore,
    fire,
    reset,
    has: (name) => entries.has(name),
  };
};

export default createScheduler;
