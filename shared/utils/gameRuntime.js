import uuid from 'uuid';

/**
 * Scoped synchronous game runtime.
 *
 * Web keeps the default behaviour (uuid/Date/Math.random). The headless RL
 * environment supplies a runtime object with `{now(), id(kind), random()}`
 * so a single episode owns its RNG, id space and logical clock.
 *
 * The context is intentionally synchronous: every gameplay handler is a
 * synchronous thunk. Returning a Promise from `withGameRuntime` is an error,
 * because an asynchronous continuation would leak the runtime into unrelated
 * code.
 */
let currentRuntime = null;

export const ASYNC_GAME_RUNTIME = 'ASYNC_GAME_RUNTIME';

export const withGameRuntime = (runtime, callback) => {
  if (runtime == null) {
    throw new Error('withGameRuntime: runtime is required');
  }
  const previous = currentRuntime;
  currentRuntime = runtime;
  try {
    const result = callback();
    if (result && typeof result.then === 'function') {
      throw new Error(ASYNC_GAME_RUNTIME);
    }
    return result;
  } finally {
    currentRuntime = previous;
  }
};

export const getGameRuntime = () => currentRuntime;

export const gameNow = () => {
  const runtime = currentRuntime;
  return runtime ? runtime.now() : Date.now();
};

export const gameId = (kind) => {
  const runtime = currentRuntime;
  return runtime ? runtime.id(kind) : uuid.v4();
};

export const gameRandom = () => {
  const runtime = currentRuntime;
  return runtime ? runtime.random() : Math.random();
};
