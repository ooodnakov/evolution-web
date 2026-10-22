import crypto from 'crypto';
import yaml from 'yaml-js';
import {List} from 'immutable';

import schema from '../schema.json';

import {PHASE} from '~/shared/models/game/GameModel';
import {
  SettingsRecord
  , SETTINGS_DEFAULT_TIME_TURN
  , SETTINGS_DEFAULT_TIME_TRAIT
  , SETTINGS_AMBUSH_TIME
} from '~/shared/models/game/GameSettings';
import {server$gameCreateSuccess, gameDestroy} from '~/shared/actions/game';
import {gameClientToServer} from '~/shared/actions/game';
import {traitClientToServer} from '~/shared/actions/trait';
import {plantsClientToServer} from '~/shared/actions/game.plantarium';

import {createRuntime} from './runtime';
import {createCatalog} from './catalog';
import {createObservation} from './observation';
import {createActionDomain, preparationPending} from './actions';
import {
  createEnvironmentStore
  , makeRoom
  , roomCreate
  , ENV_ROOM_ID
} from './store';
import {
  computePrimary
  , transientItemsBySeat
} from './decisions';
import {entitiesOfGame} from './slots';

const ADDONS = schema.addons;
const CONFIG_KEYS = ['numPlayers', 'addons', 'halfDeck', 'maxCards', 'randomPlayers', 'maxDecisions'];
const SCENARIO_KEYS = ['phase', 'food', 'deck', 'deckPlants', 'plants', 'settings', 'players'];
const SETTINGS_KEYS = [
  'timeTurn', 'timeTraitResponse', 'timeAmbush', 'randomPlayers', 'halfDeck', 'maxCards',
  'seed', ...ADDONS.map(addon => `addon_${addon}`),
];

const REQUESTS = {
  ...gameClientToServer
  , ...traitClientToServer
  , ...plantsClientToServer
};

let ENVIRONMENT_SEQ = 0;

export class EnvironmentError extends Error {
  constructor(code, message) {
    super(message ? `${code}: ${message}` : code);
    this.name = 'EnvironmentError';
    this.code = code;
    this.userLevelError = false;
    this.rlFatal = true;
  }
}

const fail = (code, message) => {
  throw new EnvironmentError(code, message);
};

const isPlainObject = (value) => value != null && typeof value === 'object' && !Array.isArray(value);

const normalizeConfig = (rawConfig) => {
  const config = rawConfig == null ? {} : rawConfig;
  if (!isPlainObject(config)) fail('INVALID_CONFIG', 'config must be an object');
  Object.keys(config).forEach((key) => {
    if (CONFIG_KEYS.indexOf(key) < 0) fail('INVALID_CONFIG', `unknown config key ${key}`);
  });

  const numPlayers = config.numPlayers === undefined ? 2 : config.numPlayers;
  if (!Number.isInteger(numPlayers) || numPlayers < 2 || numPlayers > 8) {
    fail('INVALID_CONFIG', 'numPlayers must be an integer in [2, 8]');
  }

  const addons = config.addons === undefined ? [] : config.addons;
  if (!Array.isArray(addons) || addons.some(addon => ADDONS.indexOf(addon) < 0)) {
    fail('INVALID_CONFIG', 'addons must be a subset of the known addon flags');
  }
  if (new Set(addons).size !== addons.length) fail('INVALID_CONFIG', 'duplicate addon');

  const halfDeck = config.halfDeck === undefined ? false : config.halfDeck;
  if (typeof halfDeck !== 'boolean') fail('INVALID_CONFIG', 'halfDeck must be a boolean');

  const randomPlayers = config.randomPlayers === undefined ? false : config.randomPlayers;
  if (typeof randomPlayers !== 'boolean') fail('INVALID_CONFIG', 'randomPlayers must be a boolean');

  const maxCards = config.maxCards === undefined ? null : config.maxCards;
  if (maxCards !== null && (!Number.isInteger(maxCards) || maxCards <= 0)) {
    fail('INVALID_CONFIG', 'maxCards must be a positive integer or null');
  }

  const maxDecisions = config.maxDecisions === undefined ? 20000 : config.maxDecisions;
  if (!Number.isInteger(maxDecisions) || maxDecisions <= 0) {
    fail('INVALID_CONFIG', 'maxDecisions must be a positive integer');
  }

  return {numPlayers, addons: addons.slice(), halfDeck, maxCards, randomPlayers, maxDecisions};
};

const randomSeed = () => crypto.randomBytes(4).readUInt32BE(0);

const normalizeSeed = (seed) => {
  if (seed === undefined || seed === null) return randomSeed();
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xFFFFFFFF) {
    fail('INVALID_CONFIG', 'seed must be an unsigned 32-bit integer');
  }
  return seed;
};

/** JSON is valid YAML 1.2, so a normalized scene can be handed to the parser. */
const yamlForScenario = (scenarioObject) => JSON.stringify(scenarioObject);

const parseScenario = (scenario, config) => {
  if (typeof scenario !== 'string' || scenario.trim() === '') {
    fail('INVALID_SCENARIO', 'scenario must be a non-empty YAML string');
  }
  let parsed;
  try {
    parsed = yaml.load(scenario);
  } catch (error) {
    fail('INVALID_SCENARIO', `cannot parse YAML: ${error.message}`);
  }
  if (parsed === null || parsed === undefined) parsed = {};
  if (!isPlainObject(parsed)) fail('INVALID_SCENARIO', 'scenario root must be a mapping');
  Object.keys(parsed).forEach((key) => {
    if (SCENARIO_KEYS.indexOf(key) < 0) fail('INVALID_SCENARIO', `unknown scenario key ${key}`);
  });

  const players = parsed.players === undefined ? [] : parsed.players;
  if (!Array.isArray(players)) fail('INVALID_SCENARIO', 'players must be a list');
  if (players.length > 0 && players.length !== config.numPlayers) {
    fail('INVALID_SCENARIO', `scenario has ${players.length} players, config has ${config.numPlayers}`);
  }
  players.forEach((player) => {
    if (player != null && !isPlainObject(player)) fail('INVALID_SCENARIO', 'player entry must be a mapping');
  });

  const scenarioSettings = parsed.settings === undefined ? {} : parsed.settings;
  if (!isPlainObject(scenarioSettings)) fail('INVALID_SCENARIO', 'settings must be a mapping');
  Object.keys(scenarioSettings).forEach((key) => {
    if (SETTINGS_KEYS.indexOf(key) < 0) fail('INVALID_SCENARIO', `unknown settings key ${key}`);
  });
  if (scenarioSettings.maxPlayers !== undefined
    && scenarioSettings.maxPlayers !== config.numPlayers) {
    fail('INVALID_SCENARIO', 'settings.maxPlayers conflicts with config.numPlayers');
  }
  ADDONS.forEach((addon) => {
    const key = `addon_${addon}`;
    if (scenarioSettings[key] === undefined) return;
    if (!!scenarioSettings[key] !== (config.addons.indexOf(addon) >= 0)) {
      fail('INVALID_SCENARIO', `settings.${key} conflicts with config.addons`);
    }
  });

  const ids = new Set();
  const collectIds = (source) => {
    if (typeof source !== 'string') return;
    source.split(/[,\s]+/)
      .map(token => token.trim())
      .filter(token => /^\$[A-Za-z0-9_]+$/.test(token))
      .forEach((token) => {
        if (ids.has(token)) fail('INVALID_SCENARIO', `duplicate forced entity id ${token}`);
        ids.add(token);
      });
  };
  players.forEach((player) => {
    if (!player) return;
    collectIds(player.continent);
    collectIds(player.hand);
  });
  collectIds(parsed.plants);

  if (parsed.food !== undefined && (!Number.isInteger(parsed.food) || parsed.food < 0)) {
    fail('INVALID_SCENARIO', 'food must be a non-negative integer');
  }

  const settings = buildSettings(config, parsed.settings || {}, null);
  const normalized = {
    ...parsed
    , players: players.length === 0
      ? players
      : players
    , settings: settings.toJS()
  };
  return {yaml: yamlForScenario(normalized), settings};
};

const buildSettings = (config, overrides = {}, seed = null) => {
  const base = {
    maxPlayers: config.numPlayers
    , timeTurn: SETTINGS_DEFAULT_TIME_TURN
    , timeTraitResponse: SETTINGS_DEFAULT_TIME_TRAIT
    , timeAmbush: SETTINGS_AMBUSH_TIME
    , randomPlayers: config.randomPlayers
    , halfDeck: config.halfDeck
    , maxCards: config.maxCards
    , seed
  };
  ADDONS.forEach((addon) => {
    base[`addon_${addon}`] = config.addons.indexOf(addon) >= 0;
  });
  Object.keys(overrides).forEach((key) => {
    if (SETTINGS_KEYS.indexOf(key) < 0) return;
    if (key === 'seed' || key === 'maxPlayers' || key.indexOf('addon_') === 0) return;
    base[key] = overrides[key];
  });
  ['timeTurn', 'timeTraitResponse', 'timeAmbush'].forEach((key) => {
    if (!Number.isFinite(base[key]) || base[key] <= 0) {
      fail('INVALID_SCENARIO', `${key} must be a positive number`);
    }
  });
  return new SettingsRecord(base);
};

/** Deterministic JSON (sorted keys) for checkpoint hashes. */
const canonicalize = (value) => {
  if (value === null || value === undefined) return null;
  if (Array.isArray(value)) return value.map(canonicalize);
  if (typeof value === 'object') {
    if (typeof value.toJS === 'function') return canonicalize(value.toJS());
    const result = {};
    Object.keys(value).sort().forEach((key) => {
      result[key] = canonicalize(value[key]);
    });
    return result;
  }
  return value;
};

const hashState = (payload) => crypto
  .createHash('sha256')
  .update(JSON.stringify(canonicalize(payload)))
  .digest('hex');

const publicChoiceRow = (row) => ({
  field_kind: row.field_kind
  , choice_kind: row.choice_kind
  , type_id: row.type_id
  , entity_kind: row.entity_kind
  , entity_slot: row.entity_slot
  , trait_slot: row.trait_slot
  , scalar: row.scalar
});

const reserveGameIds = (runtime, game) => {
  if (!game) return;
  const reserve = (id) => {
    if (typeof id === 'string' && id.length > 0) runtime.reserve(id);
  };
  reserve(game.id);
  game.players.forEach((player) => {
    reserve(player.id);
    player.hand.forEach(card => reserve(card.id));
    player.continent.forEach((animal) => {
      reserve(animal.id);
      animal.traits.forEach(trait => reserve(trait.id));
    });
  });
  game.plants.forEach((plant) => {
    reserve(plant.id);
    plant.traits.forEach(trait => reserve(trait.id));
  });
  game.deck.forEach(card => reserve(card.id));
  game.deckPlants && game.deckPlants.forEach(card => reserve(card && card.id));
};

export class EvolutionEnvironment {
  constructor(config = {}) {
    this.id = ++ENVIRONMENT_SEQ;
    this.config = normalizeConfig(config);
    this.catalog = createCatalog();
    this.observation = createObservation({catalog: this.catalog});
    this.domain = createActionDomain({catalog: this.catalog});
    this.agents = Array.from({length: this.config.numPlayers}, (_, seat) => `player_${seat}`);
    this.closed = false;
    this.snapshots = new Map();
    this.nextSnapshotHandle = 1;
    this.resetState();
  }

  resetState() {
    this.store = null;
    this.scheduler = null;
    this.runtime = null;
    this.gameId = null;
    this.seed = null;
    this.scenario = null;
    this.decision = null;
    this.decisionCounter = 0;
    this.stateVersion = 0;
    this.boundaryEpoch = 0;
    this.preparedEpoch = -1;
    this.transientWindow = null;
    this.pendingTransient = [];
    this.regenerationDone = new Set();
    this.maxDecisionsUsed = 0;
    this.terminalInfo = null;
    this.terminalWinner = null;
    this.terminalRewards = {};
    this.terminalRewardsIssued = false;
    this.replay = null;
    this.observationCache = new Map();
  }

  game() {
    return this.store ? this.store.getState().getIn(['games', this.gameId]) : null;
  }

  room() {
    return this.store ? this.store.getState().getIn(['rooms', ENV_ROOM_ID]) : null;
  }

  // --------------------------------------------------------------- reset
  reset(options = {}) {
    if (this.closed) fail('INTERNAL', 'environment is closed');
    const {seed, scenario} = options == null ? {} : options;
    const runtimeSeed = normalizeSeed(seed);

    if (this.gameId && this.store) {
      this.store.dispatch(gameDestroy(this.gameId));
    }

    this.resetState();
    this.seed = runtimeSeed;
    this.scenario = scenario === undefined || scenario === null ? null : scenario;
    this.runtime = createRuntime({seed: runtimeSeed});

    let settings;
    let scenarioYaml = null;
    if (this.scenario !== null) {
      const parsed = parseScenario(this.scenario, this.config);
      settings = parsed.settings;
      scenarioYaml = parsed.yaml;
    } else {
      settings = buildSettings(this.config, {}, null);
    }
    settings = settings.set('seed', scenarioYaml);

    const room = makeRoom({
      roomId: ENV_ROOM_ID
      , users: List(this.agents)
      , settings
    });

    const built = createEnvironmentStore();
    this.store = built.store;
    this.scheduler = built.scheduler;

    this.replay = {
      schema_version: schema.schema_version
      , engine_version: schema.engine_version
      , config: {...this.config}
      , seed: runtimeSeed
      , scenario: this.scenario
      , commands: []
      , checkpoints: []
    };

    this.runtime.run(() => {
      this.store.dispatch(roomCreate(room));
      this.store.dispatch(server$gameCreateSuccess(room));
    });
    this.stateVersion += 1;

    const game = this.store.getState().get('games').first();
    if (!game) fail('ENGINE_STALLED', 'game creation produced no game');
    this.gameId = game.id;
    reserveGameIds(this.runtime, game);

    this.advance();
    return this.frame();
  }

  // ------------------------------------------------------------- decisions
  makeContext(primary, extra = {}) {
    const game = this.game();
    return {
      game
      , projection: game.toOthers(primary.actor)
      , playerId: primary.actor
      , decisionKind: primary.kind
      , question: primary.question || null
      , ambush: primary.ambush || null
      , regeneration: null
      , transient: extra.transient || null
      , revision: this.stateVersion
    };
  }

  entityRef(entityId) {
    if (!entityId) return null;
    const slots = entitiesOfGame(this.game());
    if (slots.animalById.has(entityId)) {
      return {kind: 'ANIMAL', slot: slots.animalById.get(entityId).slot};
    }
    if (slots.plantById.has(entityId)) {
      return {kind: 'PLANT', slot: slots.plantById.get(entityId).slot};
    }
    if (slots.shellById.has(entityId)) {
      return {kind: 'SHELL', slot: slots.shellById.get(entityId).slot};
    }
    return null;
  }

  maskFor(choices, prefix, context) {
    return choices.map((choice) => {
      const next = [...prefix, choice];
      try {
        if (this.domain.isComplete(next, context)) return 1;
        return this.domain.choices(next, context).length > 0 ? 1 : 0;
      } catch (error) {
        return 0;
      }
    });
  }

  publishDecision({kind, actor, prefix, choices, primary, extra}) {
    const context = this.makeContext(primary, extra || {});
    const mask = this.maskFor(choices, prefix, context);
    this.decision = {
      id: ++this.decisionCounter
      , kind
      , actor
      , prefix
      , choices
      , mask
      , context
      , primary
      , sourceAid: primary.sourceAid || null
      , targetAid: primary.targetAid || null
      , attackTraitType: primary.attackTraitType
    };
    this.observationCache.clear();
    return this.decision;
  }

  publishPrimary(primary, prefix = []) {
    if (primary.terminal) return this.publishTerminal(false);
    const context = this.makeContext(primary);
    let choices;
    try {
      choices = prefix.length === 0
        ? this.domain.rootChoices(context)
        : this.domain.choices(prefix, context);
    } catch (error) {
      fail('ENGINE_STALLED', `choice enumeration failed: ${error.message}`);
    }
    if (choices.length === 0) {
      fail('ENGINE_STALLED', `empty action domain for ${primary.kind} ${primary.actor}`);
    }
    return this.publishDecision({
      kind: primary.kind
      , actor: primary.actor
      , prefix
      , choices
      , primary
    });
  }

  publishTransientSeat(primary) {
    const window = this.transientWindow;
    if (!window || window.index >= window.seats.length) return false;
    const seat = window.seats[window.index];
    const pending = window.itemsBySeat[seat] || [];
    const actorPrimary = {...primary, kind: 'TRANSIENT', actor: seat, question: null, ambush: null};
    const context = this.makeContext(actorPrimary, {transient: {pending}});
    let choices;
    try {
      choices = this.domain.rootChoices(context);
    } catch (error) {
      fail('ENGINE_STALLED', `transient enumeration failed: ${error.message}`);
    }
    if (choices.length === 0) fail('ENGINE_STALLED', 'empty transient domain');
    this.publishDecision({
      kind: 'TRANSIENT'
      , actor: seat
      , prefix: []
      , choices
      , primary: actorPrimary
      , extra: {transient: {pending}}
    });
    return true;
  }

  publishTerminal(truncated, reason = null) {
    const game = this.game();
    this.decision = {
      id: ++this.decisionCounter
      , kind: 'TERMINAL'
      , actor: null
      , prefix: []
      , choices: []
      , mask: []
      , context: null
      , primary: {terminal: true}
      , sourceAid: null
      , targetAid: null
      , attackTraitType: -1
    };
    this.observationCache.clear();
    this.terminalInfo = {truncated: !!truncated, reason};
    this.terminalRewardsIssued = false;
    this.terminalRewards = {};
    this.terminalWinner = null;

    if (truncated || !game) return this.decision;

    this.terminalWinner = game.winnerId;
    const active = this.agents.filter(agent => {
      const player = game.getPlayer(agent);
      return !!player && player.playing;
    });
    const others = Math.max(active.length - 1, 0);
    this.agents.forEach((agent) => {
      if (active.indexOf(agent) < 0) {
        this.terminalRewards[agent] = 0;
      } else if (agent === this.terminalWinner) {
        this.terminalRewards[agent] = 1;
      } else {
        this.terminalRewards[agent] = others > 0 ? -1 / others : 0;
      }
    });
    return this.decision;
  }

  // ------------------------------------------------------------- advancing
  advance() {
    for (let guard = 0; guard < 100000; guard += 1) {
      const game = this.game();
      if (!game) fail('ENGINE_STALLED', 'no game in store');

      if (game.status.phase !== PHASE.REGENERATION) this.regenerationDone.clear();

      const primary = computePrimary({game, regenerationDone: this.regenerationDone});
      if (primary.terminal) {
        this.publishTerminal(false);
        return;
      }
      if (primary.stall) fail('ENGINE_STALLED', primary.stall);

      if (primary.regenerationEnd) {
        const fired = this.runtime.run(() => this.scheduler.fire(`Phase#Regeneration#${this.gameId}`));
        if (!fired) fail('ENGINE_STALLED', 'regeneration end callback is not armed');
        this.stateVersion += 1;
        continue;
      }
      if (primary.ambushPrepareEnd) {
        const fired = this.runtime.run(() => this.scheduler.fire(`Phase#Ambush#${this.gameId}`));
        if (!fired) fail('ENGINE_STALLED', 'ambush end callback is not armed');
        this.stateVersion += 1;
        continue;
      }

      if (game.question) {
        this.publishPrimary(primary);
        return;
      }

      if (this.pendingTransient.length > 0) {
        const drained = this.drainPendingTransient();
        if (!drained) continue;
        continue;
      }

      if ((primary.kind === 'NORMAL' || primary.kind === 'DEFENSE' || primary.kind === 'INTELLECT')
        && game.status.phase === PHASE.FEEDING) {
        if (this.transientWindow) {
          if (this.publishTransientSeat(primary)) return;
          this.transientWindow = null;
          continue;
        }
        if (this.preparedEpoch !== this.boundaryEpoch) {
          const itemsBySeat = transientItemsBySeat(game, this.agents);
          const any = this.agents.some(agent => itemsBySeat[agent].length > 0);
          this.preparedEpoch = this.boundaryEpoch;
          if (any) {
            this.transientWindow = {seats: this.agents.slice(), index: 0, itemsBySeat};
            if (this.publishTransientSeat(primary)) return;
            this.transientWindow = null;
            continue;
          }
        }
      }

      this.publishPrimary(primary);
      return;
    }
    fail('ENGINE_STALLED', 'advance loop exceeded');
  }

  // -------------------------------------------------- transient vector apply
  toggleTransient({animalId, traitId, value, ownerId}) {
    const game = this.game();
    const animal = game.locateAnimal(animalId);
    if (!animal) return 'declined';
    const trait = animal.traits.get(traitId);
    if (!trait) return 'declined';
    if (trait.value === value) return 'applied';

    const before = trait.value;
    let threw = null;
    try {
      this.runtime.run(() => this.store.dispatch(
        REQUESTS.traitActivateRequest(
          {gameId: this.gameId, sourceAid: animalId, traitId, targets: []}
          , {userId: ownerId || animal.ownerId}
        )
      ));
    } catch (error) {
      if (!error || !error.userLevelError) throw error;
      threw = error;
    }
    this.stateVersion += 1;

    const updatedGame = this.game();
    const updatedAnimal = updatedGame.locateAnimal(animalId);
    const updatedTrait = updatedAnimal ? updatedAnimal.traits.get(traitId) : null;
    const applied = !!updatedTrait && updatedTrait.value !== before;
    if (applied) return 'applied';
    if (updatedGame.question || updatedGame.status.phase === PHASE.AMBUSH) return 'pending';
    if (threw) return 'declined';
    return 'pending';
  }

  applyTransientVector(vector) {
    const ordered = vector.slice().sort((left, right) => {
      const game = this.game();
      const slots = entitiesOfGame(game);
      const leftEntry = slots.animalById.get(left.animalId);
      const rightEntry = slots.animalById.get(right.animalId);
      return (leftEntry ? leftEntry.slot : 0) - (rightEntry ? rightEntry.slot : 0);
    });

    for (let index = 0; index < ordered.length; index += 1) {
      const item = ordered[index];
      const game = this.game();
      const animal = game.locateAnimal(item.animalId);
      if (!animal) continue;
      const result = this.toggleTransient(item);
      if (result === 'pending') {
        this.pendingTransient = ordered.slice(index);
        return;
      }
    }
    this.pendingTransient = [];
  }

  drainPendingTransient() {
    const game = this.game();
    if (game.question) return false;

    const pending = this.pendingTransient;
    this.pendingTransient = [];
    for (let index = 0; index < pending.length; index += 1) {
      const item = pending[index];
      const current = this.game();
      const animal = current.locateAnimal(item.animalId);
      if (!animal) continue;
      const result = this.toggleTransient(item);
      if (result === 'pending') {
        const updated = this.game();
        this.pendingTransient = pending.slice(index);
        return !(updated.question || updated.status.phase === PHASE.AMBUSH);
      }
    }
    return true;
  }

  // ------------------------------------------------------------------ step
  step(options = {}) {
    if (this.closed) fail('INTERNAL', 'environment is closed');
    if (!this.decision) fail('INVALID_DECISION', 'no active decision');

    const {decisionId, choiceIndex} = options == null ? {} : options;
    if (!Number.isInteger(decisionId) || decisionId !== this.decision.id) {
      fail('INVALID_DECISION', `expected ${this.decision.id}, got ${decisionId}`);
    }
    const {choices, prefix, context} = this.decision;
    if (!Number.isInteger(choiceIndex) || choiceIndex < 0 || choiceIndex >= choices.length) {
      fail('INVALID_CHOICE', `index ${choiceIndex} out of range [0, ${choices.length})`);
    }

    const choice = choices[choiceIndex];
    const nextPrefix = [...prefix, choice];

    let complete = false;
    try {
      complete = this.domain.isComplete(nextPrefix, context);
    } catch (error) {
      fail('INVALID_CHOICE', error.message);
    }

    if (!complete) {
      let nextChoices;
      try {
        nextChoices = this.domain.choices(nextPrefix, context);
      } catch (error) {
        fail('INVALID_CHOICE', error.message);
      }
      if (!nextChoices || nextChoices.length === 0) {
        fail('ENGINE_STALLED', 'prefix has no continuation');
      }
      this.publishDecision({
        kind: this.decision.kind
        , actor: this.decision.actor
        , prefix: nextPrefix
        , choices: nextChoices
        , primary: this.decision.primary
      });
      return this.frame();
    }

    return this.completeDecision(nextPrefix, context);
  }

  completeDecision(prefix, context) {
    const decision = this.decision;
    const command = this.domain.resolveCommand(prefix, context);
    const actor = decision.actor;
    const kind = decision.kind;
    const publicChoices = prefix.map(publicChoiceRow);

    if (command.special) {
      return this.applySpecial(command, {actor, kind, publicChoices});
    }

    const request = {type: command.type, ...command.data};
    const preparation = preparationPending(request, context);

    const core = this.captureCore();
    let failure = null;
    try {
      this.runtime.run(() => this.store.dispatch(
        REQUESTS[command.type]({gameId: this.gameId, ...command.data}, {userId: actor})
      ));
    } catch (error) {
      failure = error;
    }

    if (failure) {
      this.applyCore(core);
      if (!failure.userLevelError) {
        if (process.env.RL_DEBUG) console.error(failure.stack);
        fail('ENGINE_INVARIANT', failure.message);
      }

      let publiclyInvalid = false;
      try {
        // eslint-disable-next-line global-require
        const {validateGameRequest} = require('~/shared/actions/game-validation');
        validateGameRequest(this.game().toOthers(actor), actor, request, 'complete');
      } catch (validationError) {
        publiclyInvalid = !!(validationError && validationError.userLevelError);
      }

      if (!preparation && publiclyInvalid) {
        fail('ENGINE_INVARIANT', `enumerator admitted a publicly invalid ${command.type}`);
      }

      this.replay.commands.push({
        decision_id: decision.id
        , actor
        , kind
        , type: command.type
        , choices: publicChoices
        , result: 'attempt_rejected'
      });
      this.maxDecisionsUsed += 1;
      this.publishPrimary(decision.primary, []);
      if (this.shouldTruncate()) return this.truncate();
      return this.frame({attemptRejected: true});
    }

    this.stateVersion += 1;
    this.runtime.tick();
    this.boundaryEpoch += 1;
    this.regenerationDone.clear();
    this.maxDecisionsUsed += 1;

    this.replay.commands.push({
      decision_id: decision.id
      , actor
      , kind
      , type: command.type
      , choices: publicChoices
      , result: 'committed'
    });
    this.replay.checkpoints.push({decision_id: this.decision.id, hash: this.stateHash()});

    this.advance();
    if (this.shouldTruncate()) return this.truncate();
    return this.frame({committed: true});
  }

  applySpecial(command, meta) {
    if (command.special === 'USE_DEFAULT') {
      this.regenerationDone.add(meta.actor);
      this.replay.commands.push({...meta, type: 'USE_DEFAULT', result: 'committed'});
      this.advance();
      if (this.shouldTruncate()) return this.truncate();
      return this.frame({committed: true});
    }

    if (command.special === 'TRANSIENT') {
      this.replay.commands.push({...meta, type: 'TRANSIENT', result: 'committed'});
      this.applyTransientVector(command.vector || []);
      if (this.transientWindow) this.transientWindow.index += 1;
      this.advance();
      if (this.shouldTruncate()) return this.truncate();
      return this.frame({committed: true});
    }

    if (command.special === 'READY') {
      this.replay.commands.push({...meta, type: 'READY', result: 'committed'});
      if (this.transientWindow) this.transientWindow.index += 1;
      this.advance();
      if (this.shouldTruncate()) return this.truncate();
      return this.frame({committed: true});
    }

    fail('ENGINE_INVARIANT', `unknown special command ${command.special}`);
  }

  shouldTruncate() {
    return !!this.decision
      && this.decision.kind !== 'TERMINAL'
      && this.maxDecisionsUsed >= this.config.maxDecisions;
  }

  truncate() {
    this.publishTerminal(true, 'max_decisions');
    return this.frame();
  }

  // --------------------------------------------------------------- observing
  viewFor(agent) {
    const decision = this.decision;
    if (!decision) return {actor: null, decisionKind: 'TERMINAL', choiceRows: [], mask: [], prefix: []};
    const isActor = decision.actor === agent;
    return {
      actor: decision.actor
      , decisionKind: decision.kind
      , choiceRows: isActor ? decision.choices : []
      , mask: isActor ? decision.mask : []
      , source: isActor ? this.entityRef(decision.sourceAid) : null
      , target: isActor ? this.entityRef(decision.targetAid) : null
      , attackTraitType: isActor ? decision.attackTraitType : -1
      , prefix: isActor ? decision.prefix : []
    };
  }

  observeFor(agent) {
    const cached = this.observationCache.get(agent);
    if (cached) return cached;
    const view = this.observation.observe(this.game(), agent, this.viewFor(agent));
    this.observationCache.set(agent, view);
    return view;
  }

  observe(agent) {
    if (this.agents.indexOf(agent) < 0) fail('INVALID_CHOICE', `unknown agent ${agent}`);
    if (!this.store) fail('ENGINE_STALLED', 'environment is not reset');
    return this.observeFor(agent);
  }

  // ------------------------------------------------------------------ frame
  frame(result = {}) {
    const observations = {};
    const rewards = {};
    const terminated = {};
    const truncated = {};
    const info = {};
    const terminal = !!this.decision && this.decision.kind === 'TERMINAL';
    const issueTerminal = terminal && !this.terminalRewardsIssued;

    this.agents.forEach((agent) => {
      observations[agent] = this.observeFor(agent);
      rewards[agent] = issueTerminal ? (this.terminalRewards[agent] || 0) : 0;
      terminated[agent] = terminal ? !this.terminalInfo.truncated : false;
      truncated[agent] = terminal ? !!this.terminalInfo.truncated : false;
      info[agent] = {
        committed: !!result.committed
        , attempt_rejected: !!result.attemptRejected
        , truncation_reason: truncated[agent] ? this.terminalInfo.reason : null
        , outcome: terminal && !this.terminalInfo.truncated ? {winner: this.terminalWinner} : null
      };
    });

    if (issueTerminal) this.terminalRewardsIssued = true;

    return {
      actor: this.decision ? this.decision.actor : null
      , decisionId: this.decision ? this.decision.id : this.decisionCounter
      , observations
      , rewards
      , terminated
      , truncated
      , info
      , committed: !!result.committed
    };
  }

  // ----------------------------------------------------- snapshot / restore
  captureCore() {
    return {
      state: this.store ? this.store.getState() : null
      , runtime: this.runtime ? this.runtime.capture() : null
      , scheduler: this.scheduler ? this.scheduler.capture() : null
      , decisionCounter: this.decisionCounter
      , stateVersion: this.stateVersion
      , boundaryEpoch: this.boundaryEpoch
      , preparedEpoch: this.preparedEpoch
      , transientWindow: this.transientWindow
        ? {
          seats: this.transientWindow.seats.slice()
          , index: this.transientWindow.index
          , itemsBySeat: this.transientWindow.itemsBySeat
        }
        : null
      , pendingTransient: this.pendingTransient.slice()
      , regenerationDone: Array.from(this.regenerationDone)
      , maxDecisionsUsed: this.maxDecisionsUsed
      , terminalInfo: this.terminalInfo
      , terminalWinner: this.terminalWinner
      , terminalRewards: {...this.terminalRewards}
      , terminalRewardsIssued: this.terminalRewardsIssued
      , gameId: this.gameId
      , seed: this.seed
      , scenario: this.scenario
    };
  }

  applyCore(core) {
    const built = createEnvironmentStore(core.state);
    this.store = built.store;
    this.scheduler = built.scheduler;
    this.scheduler.restore(core.scheduler);
    this.runtime = createRuntime({seed: core.runtime.seed});
    this.runtime.restore(core.runtime);
    this.decisionCounter = core.decisionCounter;
    this.stateVersion = core.stateVersion;
    this.boundaryEpoch = core.boundaryEpoch;
    this.preparedEpoch = core.preparedEpoch;
    this.transientWindow = core.transientWindow
      ? {
        seats: core.transientWindow.seats.slice()
        , index: core.transientWindow.index
        , itemsBySeat: core.transientWindow.itemsBySeat
      }
      : null;
    this.pendingTransient = core.pendingTransient.slice();
    this.regenerationDone = new Set(core.regenerationDone);
    this.maxDecisionsUsed = core.maxDecisionsUsed;
    this.terminalInfo = core.terminalInfo;
    this.terminalWinner = core.terminalWinner;
    this.terminalRewards = {...core.terminalRewards};
    this.terminalRewardsIssued = core.terminalRewardsIssued;
    this.gameId = core.gameId;
    this.seed = core.seed;
    this.scenario = core.scenario;
    this.decision = null;
    this.observationCache.clear();
  }

  capture() {
    return {
      ...this.captureCore()
      , decision: this.decision
        ? {id: this.decision.id, kind: this.decision.kind, actor: this.decision.actor, prefix: this.decision.prefix.slice(), primary: this.decision.primary}
        : null
      , replayLength: this.replay ? this.replay.commands.length : 0
    };
  }

  snapshot() {
    if (this.closed) fail('INVALID_SNAPSHOT', 'environment is closed');
    if (!this.decision) fail('INVALID_SNAPSHOT', 'no decision boundary');
    const handle = this.nextSnapshotHandle++;
    this.snapshots.set(handle, this.capture());
    return handle;
  }

  restore(handle) {
    if (this.closed) fail('INVALID_SNAPSHOT', 'environment is closed');
    if (!Number.isInteger(handle) || !this.snapshots.has(handle)) {
      fail('INVALID_SNAPSHOT', `unknown snapshot handle ${handle}`);
    }
    const captured = this.snapshots.get(handle);
    this.applyCore(captured);
    if (this.replay) this.replay.commands.length = captured.replayLength;

    const saved = captured.decision;
    if (!saved) {
      this.publishTerminal(!!this.terminalInfo && !!this.terminalInfo.truncated, this.terminalInfo ? this.terminalInfo.reason : null);
      return this.frame();
    }

    if (saved.kind === 'TERMINAL') {
      this.publishTerminal(!!this.terminalInfo && !!this.terminalInfo.truncated, this.terminalInfo ? this.terminalInfo.reason : null);
      return this.frame();
    }

    this.publishPrimary(saved.primary, saved.prefix);
    return this.frame();
  }

  exportReplay() {
    if (!this.replay) fail('INTERNAL', 'environment is not reset');
    return JSON.parse(JSON.stringify(this.replay));
  }

  stateHash() {
    const game = this.game();
    return hashState({
      game: game ? game.toJS() : null
      , runtime: this.runtime ? this.runtime.capture() : null
      , counters: {
        boundaryEpoch: this.boundaryEpoch
        , maxDecisions: this.maxDecisionsUsed
      }
    });
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    this.snapshots.clear();
    this.store = null;
    this.scheduler = null;
    this.runtime = null;
    this.decision = null;
    this.observationCache.clear();
  }
}

export const createEnvironment = (config = {}) => new EvolutionEnvironment(config);

export default createEnvironment;
