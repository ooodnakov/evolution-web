import {withGameRuntime} from '~/shared/utils/gameRuntime';

/**
 * Episode runtime for the headless RL environment.
 *
 * Owns the three pieces of global state the gameplay code needs:
 * seeded RNG (Mulberry32), a monotonic entity id counter and a logical clock.
 * No native timers, `Date.now` or `Math.random` are used, so two runtimes can
 * be interleaved in one process without sharing state.
 */

const UINT32_MAX = 0xFFFFFFFF;

const isUint32 = (value) => typeof value === 'number'
  && Number.isInteger(value)
  && value >= 0
  && value <= UINT32_MAX;

const isCounter = (value) => typeof value === 'number'
  && Number.isSafeInteger(value)
  && value >= 0;

const assertSeed = (seed) => {
  if (!isUint32(seed)) {
    throw new Error('INVALID_SEED');
  }
};

const asPlainObject = (value) => value != null && typeof value === 'object' && !Array.isArray(value);

export const createRuntime = (options = {}) => {
  const {seed} = options == null ? {} : options;
  assertSeed(seed);

  let seedValue = seed;
  let rngState = seed >>> 0;
  let idCounter = 0;
  let clock = 0;
  let reserved = new Set();

  // Mulberry32. Single uint32 of state; advance-then-mix.
  const random = () => {
    let t = (rngState = (rngState + 0x6D2B79F5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const nextId = (kind) => {
    if (typeof kind !== 'string' || kind === '') {
      throw new Error('INVALID_ID_KIND');
    }
    let candidate;
    do {
      candidate = `${kind}_${++idCounter}`;
    } while (reserved.has(candidate));
    return candidate;
  };

  const runtime = {
    get seed() {
      return seedValue;
    },
    random,
    nextId,
    // Alias used by `withGameRuntime`/`gameId`.
    id: nextId,
    reserve(id) {
      if (typeof id !== 'string' || id === '') {
        throw new Error('INVALID_ID');
      }
      reserved.add(id);
      return id;
    },
    hasId(id) {
      return reserved.has(id);
    },
    now() {
      return clock;
    },
    tick() {
      clock += 1;
      return clock;
    },
    run(callback) {
      return withGameRuntime(runtime, callback);
    },
    capture() {
      return {
        seed: seedValue,
        rngState,
        idCounter,
        clock,
        reserved: Array.from(reserved),
      };
    },
    restore(state) {
      if (!asPlainObject(state)) {
        throw new Error('INVALID_RUNTIME_STATE');
      }
      const {
        seed: capturedSeed,
        rngState: capturedRngState,
        idCounter: capturedIdCounter,
        clock: capturedClock,
        reserved: capturedReserved,
      } = state;
      if (!isUint32(capturedSeed)
        || !isUint32(capturedRngState)
        || !isCounter(capturedIdCounter)
        || !isCounter(capturedClock)
        || !Array.isArray(capturedReserved)
        || !capturedReserved.every((id) => typeof id === 'string' && id !== '')) {
        throw new Error('INVALID_RUNTIME_STATE');
      }
      seedValue = capturedSeed;
      rngState = capturedRngState;
      idCounter = capturedIdCounter;
      clock = capturedClock;
      reserved = new Set(capturedReserved);
    },
  };

  return runtime;
};

export default createRuntime;
