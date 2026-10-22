import {getGameRuntime} from './gameRuntime';

const originalRandomFn = (min, max) => Math.floor(min + Math.random() * (max + 1 - min));

let randomFn = originalRandomFn;

export const getIntRandom = (min, max) => {
  const runtime = getGameRuntime();
  // Inclusive [min, max]. The runtime RNG replaces Math.random entirely so an
  // episode reproduces its dice; the no-runtime branch keeps web/test behaviour.
  return runtime
    ? Math.floor(min + runtime.random() * (max + 1 - min))
    : randomFn(min, max);
};

export const getRandom = () => {
  const runtime = getGameRuntime();
  return runtime ? runtime.random() : Math.random();
};

// TODO remove this shit and rewrite to proper sinon stub =/

export const replaceGetRandom = (randomCb, testCb) => {
  randomFn = randomCb;
  testCb();
  randomFn = originalRandomFn;
};
