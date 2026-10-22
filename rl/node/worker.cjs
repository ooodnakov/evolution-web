#!/usr/bin/env node
'use strict';

/*
 * Headless JSONL transport for the Evolution RL engine.
 *
 * Contract: rl/schema.json -> "protocol".
 *
 * stdin  : one JSON request per line  {id:int, op:string, payload:object}
 * stdout : one JSON response per line {id:int, ok:true,  result:object}
 *                                    {id:int, ok:false, error:{code,message}}
 *
 * stdout carries protocol frames only. Anything human-readable goes to
 * stderr (LOG_STREAM=stderr is set before any application module loads).
 */

process.env.LOG_STREAM = 'stderr';

const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const SCHEMA_PATH = path.join(REPO_ROOT, 'rl', 'schema.json');

// Never let an accidental console.log() corrupt the JSONL stream.
console.log = console.error;
console.info = console.error;

const SCHEMA = JSON.parse(fs.readFileSync(SCHEMA_PATH, 'utf8'));
const ERROR_CODES = new Set(SCHEMA.protocol.error_codes);
const CONFIG_KEYS = ['numPlayers', 'addons', 'halfDeck', 'maxCards', 'randomPlayers', 'maxDecisions'];
const KNOWN_ADDONS = new Set(SCHEMA.addons);

function loadBabelRegister() {
  const babelrcPath = path.join(REPO_ROOT, '.babelrc');
  let babelrc = { presets: [], plugins: [] };
  if (fs.existsSync(babelrcPath)) {
    babelrc = JSON.parse(fs.readFileSync(babelrcPath, 'utf8'));
  }
  require('@babel/register')({
    cwd: REPO_ROOT,
    // The same .babelrc the web build uses, applied explicitly so the
    // worker never depends on its own cwd for config discovery.
    babelrc: false,
    configFile: false,
    presets: babelrc.presets || [],
    plugins: babelrc.plugins || [],
    extensions: ['.js'],
    ignore: [/node_modules/],
  });
}

function makeError(code, message) {
  const error = new Error(message);
  error.code = code;
  return error;
}

function invalidConfig(message) {
  throw makeError('INVALID_CONFIG', message);
}

function asPayload(payload) {
  if (payload === undefined || payload === null) return {};
  if (typeof payload !== 'object' || Array.isArray(payload)) {
    throw makeError('INTERNAL', 'payload must be a JSON object');
  }
  return payload;
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function normalizeConfig(input) {
  if (input === undefined || input === null) input = {};
  if (!isPlainObject(input)) invalidConfig('config must be an object');

  const unknown = Object.keys(input).filter((key) => !CONFIG_KEYS.includes(key));
  if (unknown.length > 0) {
    invalidConfig('unknown config keys: ' + unknown.join(','));
  }

  const defaults = SCHEMA.config;
  const numPlayers = input.numPlayers === undefined ? defaults.numPlayers : input.numPlayers;
  if (!Number.isInteger(numPlayers) || numPlayers < 2 || numPlayers > 8) {
    invalidConfig('numPlayers must be an integer between 2 and 8');
  }

  const rawAddons = input.addons === undefined ? defaults.addons : input.addons;
  if (!Array.isArray(rawAddons)) invalidConfig('addons must be an array of strings');
  const addons = [];
  for (const addon of rawAddons) {
    if (typeof addon !== 'string' || !KNOWN_ADDONS.has(addon)) {
      invalidConfig('unknown addon: ' + String(addon));
    }
    if (!addons.includes(addon)) addons.push(addon);
  }

  const halfDeck = input.halfDeck === undefined ? defaults.halfDeck : input.halfDeck;
  if (typeof halfDeck !== 'boolean') invalidConfig('halfDeck must be a boolean');

  const maxCards = input.maxCards === undefined ? defaults.maxCards : input.maxCards;
  if (maxCards !== null && (!Number.isInteger(maxCards) || maxCards <= 0)) {
    invalidConfig('maxCards must be null or a positive integer');
  }

  const randomPlayers = input.randomPlayers === undefined ? defaults.randomPlayers : input.randomPlayers;
  if (typeof randomPlayers !== 'boolean') invalidConfig('randomPlayers must be a boolean');

  const maxDecisions = input.maxDecisions === undefined ? defaults.maxDecisions : input.maxDecisions;
  if (!Number.isInteger(maxDecisions) || maxDecisions <= 0) {
    invalidConfig('maxDecisions must be a positive integer');
  }

  // Canonical key order keeps deep-equality cheap for repeated hello calls.
  return { numPlayers, addons, halfDeck, maxCards, randomPlayers, maxDecisions };
}

function validateSeed(seed) {
  if (seed === undefined || seed === null) return null;
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw makeError('INVALID_CONFIG', 'seed must be null or a uint32 integer');
  }
  return seed;
}

function validateScenario(scenario) {
  if (scenario === undefined || scenario === null) return null;
  if (typeof scenario !== 'string') {
    throw makeError('INVALID_SCENARIO', 'scenario must be null or a YAML string');
  }
  return scenario;
}

function loadEnvironmentModule() {
  // Loaded lazily so that the "before hello" error path never touches the
  // gameplay graph; @babel/register must be installed first.
  loadBabelRegister();
  return require('./environment.js');
}

function resolveCreateEnvironment(mod) {
  const candidate =
    (mod && typeof mod.createEnvironment === 'function' && mod.createEnvironment) ||
    (mod && mod.default && typeof mod.default.createEnvironment === 'function'
      ? mod.default.createEnvironment
      : null) ||
    (mod && typeof mod.default === 'function' ? mod.default : null);
  if (typeof candidate !== 'function') {
    throw makeError('INTERNAL', 'environment.js does not export createEnvironment');
  }
  return candidate;
}

function loadCatalog() {
  try {
    const mod = require('./catalog.js');
    if (typeof mod === 'function') return mod();
    if (mod && typeof mod.getCatalog === 'function') return mod.getCatalog();
    if (mod && mod.catalog !== undefined) return mod.catalog;
    if (mod && mod.CATALOG !== undefined) return mod.CATALOG;
    if (mod && mod.default !== undefined) {
      return typeof mod.default === 'function' ? mod.default() : mod.default;
    }
    return {};
  } catch (error) {
    const message = String((error && error.message) || '');
    if (
      error &&
      error.code === 'MODULE_NOT_FOUND' &&
      /Cannot find module '\.\/catalog\.js'/.test(message)
    ) {
      // catalog.js is owned by the observation slice; it may not exist yet.
      return {};
    }
    throw error;
  }
}

const state = {
  env: null,
  config: null,
  catalog: null,
  closeRequested: false,
  exiting: false,
};

function helloResult() {
  return {
    protocol_version: SCHEMA.protocol_version,
    schema_version: SCHEMA.schema_version,
    engine_version: SCHEMA.engine_version,
    catalog: state.catalog,
    config: state.config,
  };
}

function requireEnv() {
  if (!state.env) throw makeError('NOT_INITIALIZED', 'hello must be called first');
  return state.env;
}

function normalizeFrame(frame) {
  if (!isPlainObject(frame)) {
    throw makeError('INTERNAL', 'engine returned a non-object Frame');
  }
  return frame;
}

function handleRequest(op, payload) {
  switch (op) {
    case 'hello': {
      const config = normalizeConfig(payload.config);
      if (state.env) {
        if (JSON.stringify(state.config) !== JSON.stringify(config)) {
          invalidConfig('hello with a different config on the same process');
        }
        return helloResult();
      }
      const createEnvironment = resolveCreateEnvironment(loadEnvironmentModule());
      state.env = createEnvironment(config);
      state.config = config;
      state.catalog = loadCatalog();
      return helloResult();
    }
    case 'reset': {
      const env = requireEnv();
      const seed = validateSeed(payload.seed);
      const scenario = validateScenario(payload.scenario);
      return normalizeFrame(env.reset({ seed, scenario }));
    }
    case 'observe': {
      const env = requireEnv();
      if (typeof payload.agent !== 'string' || payload.agent.length === 0) {
        throw makeError('INTERNAL', 'observe requires a string agent');
      }
      return env.observe(payload.agent);
    }
    case 'step': {
      const env = requireEnv();
      const decisionId = payload.decision_id !== undefined ? payload.decision_id : payload.decisionId;
      const choiceIndex =
        payload.choice_index !== undefined ? payload.choice_index : payload.choiceIndex;
      if (!Number.isInteger(decisionId)) {
        throw makeError('INVALID_DECISION', 'decision_id must be an integer');
      }
      if (!Number.isInteger(choiceIndex) || choiceIndex < 0) {
        throw makeError('INVALID_CHOICE', 'choice_index must be a non-negative integer');
      }
      return normalizeFrame(env.step({ decisionId, choiceIndex }));
    }
    case 'snapshot': {
      const env = requireEnv();
      return { handle: env.snapshot() };
    }
    case 'restore': {
      const env = requireEnv();
      if (!Number.isInteger(payload.handle)) {
        throw makeError('INVALID_SNAPSHOT', 'handle must be an integer');
      }
      return normalizeFrame(env.restore(payload.handle));
    }
    case 'export_replay': {
      const env = requireEnv();
      return env.exportReplay();
    }
    case 'close': {
      state.closeRequested = true;
      if (state.env) {
        try {
          state.env.close();
        } finally {
          state.env = null;
        }
      }
      return { closed: true };
    }
    default:
      throw makeError('INTERNAL', 'unknown op: ' + String(op));
  }
}

function errorResponse(id, error) {
  const code =
    error && typeof error.code === 'string' && ERROR_CODES.has(error.code)
      ? error.code
      : 'INTERNAL';
  const message =
    error && typeof error.message === 'string' && error.message.length > 0
      ? error.message
      : String(error);
  return { id, ok: false, error: { code, message } };
}

function writeResponse(response, done) {
  let payload;
  try {
    payload = JSON.stringify(response) + '\n';
  } catch (error) {
    // Engine values must stay JSON-safe (no BigInt / cycles); report the
    // failure for this request without corrupting the stream.
    const fallback = errorResponse(
      response && Number.isInteger(response.id) ? response.id : null,
      makeError('INTERNAL', 'engine result is not JSON-serialisable: ' + error.message)
    );
    payload = JSON.stringify(fallback) + '\n';
  }
  if (typeof done === 'function') {
    process.stdout.write(payload, done);
  } else {
    process.stdout.write(payload);
  }
}

function processLine(line) {
  const trimmed = line.trim();
  if (trimmed.length === 0) return null;

  let request;
  try {
    request = JSON.parse(trimmed);
  } catch (error) {
    return errorResponse(null, makeError('INTERNAL', 'malformed JSON request: ' + error.message));
  }

  if (!isPlainObject(request)) {
    return errorResponse(null, makeError('INTERNAL', 'request must be a JSON object'));
  }
  const id = Number.isInteger(request.id) ? request.id : null;

  try {
    const result = handleRequest(request.op, asPayload(request.payload));
    return { id, ok: true, result: result === undefined ? {} : result };
  } catch (error) {
    return errorResponse(id, error);
  }
}

function closeEnvQuietly() {
  if (!state.env) return;
  try {
    state.env.close();
  } catch (error) {
    try {
      process.stderr.write('worker close failed: ' + String((error && error.stack) || error) + '\n');
    } catch (ignored) {
      // nothing sensible left to do while shutting down
    }
  } finally {
    state.env = null;
  }
}

function shutdown(code) {
  if (state.exiting) return;
  state.exiting = true;
  closeEnvQuietly();
  process.exit(code);
}

function finishAfterFlush(code) {
  if (state.exiting) return;
  state.exiting = true;
  closeEnvQuietly();
  process.exit(code);
}

function runJsonlLoop() {
  const readline = require('readline');
  const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  let pending = Promise.resolve();

  rl.on('line', (line) => {
    pending = pending.then(() => {
      const response = processLine(line);
      const exitAfter = state.closeRequested;
      if (exitAfter) {
        // Mark exiting before closing readline so its 'close' handler does
        // not race the pending stdout flush.
        state.exiting = true;
        if (response) {
          writeResponse(response, () => finishAfterFlush(0));
        } else {
          finishAfterFlush(0);
        }
        rl.close();
        return;
      }
      if (response) writeResponse(response);
    });
  });

  rl.on('close', () => {
    pending
      .then(() => {
        if (state.exiting) return;
        closeEnvQuietly();
        process.exit(0);
      })
      .catch((error) => {
        try {
          process.stderr.write('worker fatal: ' + String((error && error.stack) || error) + '\n');
        } catch (ignored) {
          // stderr unavailable
        }
        closeEnvQuietly();
        process.exit(1);
      });
  });
}

function frameIsTerminated(frame) {
  if (!isPlainObject(frame)) return false;
  const terminated = frame.terminated;
  if (!isPlainObject(terminated)) return false;
  const agents = Object.keys(terminated);
  if (agents.length === 0) return false;
  return agents.every((agent) => terminated[agent] === true);
}

function runSmoke() {
  const report = {
    ok: false,
    decisions: 0,
    terminated: false,
    truncated: false,
    frame: null,
  };
  let env = null;
  let exitCode = 1;
  try {
    const createEnvironment = resolveCreateEnvironment(loadEnvironmentModule());
    const config = normalizeConfig({ numPlayers: 2 });
    env = createEnvironment(config);
    let frame = normalizeFrame(env.reset({ seed: 0, scenario: null }));
    let decisions = 0;
    const maxDecisions = config.maxDecisions;

    while (frame.actor !== null && frame.actor !== undefined && decisions < maxDecisions) {
      const decisionId =
        frame.decisionId !== undefined ? frame.decisionId : frame.decision_id;
      if (!Number.isInteger(decisionId)) {
        throw makeError('INTERNAL', 'engine Frame is missing an integer decisionId');
      }
      frame = normalizeFrame(env.step({ decisionId, choiceIndex: 0 }));
      decisions += 1;
    }

    const terminated = frameIsTerminated(frame);
    const truncated = !terminated && decisions >= maxDecisions;
    report.decisions = decisions;
    report.terminated = terminated;
    report.truncated = truncated;
    report.frame = frame;
    report.ok = terminated;
    exitCode = terminated ? 0 : 1;
  } catch (error) {
    report.error = {
      code:
        error && typeof error.code === 'string' && ERROR_CODES.has(error.code)
          ? error.code
          : 'INTERNAL',
      message: String((error && error.message) || error),
    };
    report.ok = false;
    exitCode = 1;
  } finally {
    if (env) {
      try {
        env.close();
      } catch (error) {
        try {
          process.stderr.write('smoke close failed: ' + String((error && error.stack) || error) + '\n');
        } catch (ignored) {
          // stderr unavailable
        }
      }
    }
  }

  process.stdout.write(JSON.stringify(report) + '\n', () => {
    process.exit(exitCode);
  });
}

function main() {
  process.on('SIGINT', () => shutdown(0));
  process.on('SIGTERM', () => shutdown(0));

  if (process.argv.includes('--smoke')) {
    runSmoke();
    return;
  }
  runJsonlLoop();
}

main();

